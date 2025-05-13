import os
import json
import logging
import sqlite3
import time
import random
from pathlib import Path
import pandas as pd
import numpy as np
import folium
from folium.plugins import MarkerCluster, FastMarkerCluster
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderUnavailable
from sklearn.cluster import KMeans
import jinja2
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

# fuck print just log

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("advisor_map.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("advisor_map")

CONFIG = {
    'input': {
        'heartland_csv': 'HEARTLAND.csv',
        'locations_csv': 'LOCATIONUSA.csv',
    },
    'output': {
        'dir': 'maps',
        'map_filename': 'geomapbyFlame.html',
        'advisors_export': 'final_advisors_w_clusters.csv'
    },
    'clustering': {
        'n_clusters': 5,
        'random_state': 42,
        'colors': ['red', 'blue', 'green', 'purple', 'orange']
    },
    'geocoding': {
        'cache_db': 'geocode_cache.sqlite',
        'user_agent': 'interactive_map_geocoder_v3',
        'timeout': 10,
        'max_retries': 3,
        'rate_limit': 1.0  # change if u want, seconds between requests
    },
    'map': {
        'default_center': [39.8283, -98.5795],
        'default_zoom': 6
    },
    'opportunity': {
        'high_value_threshold': 10000000,
        'opportunity_per_advisor_threshold': 5000000
    }
}

class GeocodingManager:
    def __init__(self, config=None):
        if config is None:
            config = CONFIG['geocoding']
            
        self.config = config
        self.cache_db = config['cache_db']
        self.geolocator = Nominatim(user_agent=config['user_agent'])
        self._init_cache()
        self.last_request_time = 0
        logger.info(f"[*] Starting geocoding manager with cache: {self.cache_db}")
    def _init_cache(self):
        conn = sqlite3.connect(self.cache_db)
        cursor = conn.cursor()
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS geocode_cache (
            key TEXT PRIMARY KEY,
            latitude REAL,
            longitude REAL,
            timestamp INTEGER,
            success BOOLEAN
        )
        ''')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_geocode_key ON geocode_cache(key)')
        conn.commit()
        conn.close()
    
# connection pooling

    def get_coordinates(self, city, state, country="USA"):
        cache_key = f"{str(city).strip().upper()}_{state}"
        conn = sqlite3.connect(self.cache_db, timeout=30.0)
        try:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT latitude, longitude, success FROM geocode_cache WHERE key = ? LIMIT 1", 
                (cache_key,)
            )
            result = cursor.fetchone()
            if result:
                if result[2]:
                    return (result[0], result[1])
                else:
                    return None
            coords = self._geocode_with_rate_limit(city, state, country)
            success = coords is not None
            cursor.execute(
                "INSERT OR REPLACE INTO geocode_cache VALUES (?, ?, ?, ?, ?)",
                (
                    cache_key, 
                    coords[0] if coords else None, 
                    coords[1] if coords else None, 
                    int(time.time()),
                    success
                )
            )
            conn.commit()
            return coords
        except sqlite3.Error as e:
            logger.error(f"Oops!: {e}")
            return None
        finally:
            conn.close()

# so u dont get rekt

    def _geocode_with_rate_limit(self, city, state, country="USA"):
        now = time.time()
        elapsed = now - self.last_request_time
        sleep_time = max(0, self.config['rate_limit'] - elapsed)
        if sleep_time > 0:
            time.sleep(sleep_time)
        location_str = f"{city}, {state}, {country}"
        for attempt in range(self.config['max_retries']):
            try:
                self.last_request_time = time.time()
                location = self.geolocator.geocode(
                    location_str, 
                    timeout=self.config['timeout'],
                    exactly_one=True 
                )
                if location:
                    return (location.latitude, location.longitude)
                return None
            except (GeocoderTimedOut, GeocoderUnavailable) as e:
                logger.warning(f"Oops! Geocoding attempt {attempt+1} failed for {location_str}: {e}")
                if attempt == self.config['max_retries'] - 1:
                    logger.error(f"Oops! All geocoding attempts failed for {location_str}")
                    return None
                base_delay = 2 ** attempt
                jitter = random.uniform(0, 0.1 * base_delay)
                time.sleep(base_delay + jitter)
        return None
    
    def batch_geocode(self, locations_df, city_col='Office City', state_col='Office State'):
        if locations_df.empty:
            return pd.DataFrame()
            
        logger.info(f"[*] Starting batch geocoding for {len(locations_df)} locations")
        
        unique_locations = (
            locations_df[[city_col, state_col]]
            .drop_duplicates()
            .dropna()
            .assign(
                city_upper=lambda df: df[city_col].str.strip().str.upper(),
                location_key=lambda df: df['city_upper'] + '_' + df[state_col]
            )
        )
        
        total_locations = len(unique_locations)
        logger.info(f"[*] Found {total_locations} unique locations to process")
        
        conn = sqlite3.connect(self.cache_db)
        try:
            conn.execute("CREATE TEMPORARY TABLE IF NOT EXISTS temp_locations (key TEXT PRIMARY KEY)")
            
            location_keys = unique_locations['location_key'].tolist()
            conn.executemany(
                "INSERT OR IGNORE INTO temp_locations VALUES (?)", 
                [(key,) for key in location_keys]
            )
            cursor = conn.execute("""
                SELECT t.key, c.latitude, c.longitude, c.success 
                FROM temp_locations t
                LEFT JOIN geocode_cache c ON t.key = c.key
                WHERE c.key IS NOT NULL
            """)
            
            cached_results = {row[0]: (row[1], row[2]) if row[3] else None for row in cursor.fetchall()}
            logger.info(f"[*] Found {len(cached_results)} locations in cache")
            
            conn.execute("DROP TABLE temp_locations")
            conn.commit()
        
        except sqlite3.Error as e:
            logger.error(f"Oops!: {e}")
            cached_results = {}
        finally:
            conn.close()
        
        coord_dict = {}
        uncached_count = 0
        
        for idx, row in unique_locations.iterrows():
            city, state = row[city_col], row[state_col]
            key = (str(city).strip().upper(), state)
            location_key = row['location_key']
            
            if location_key in cached_results:
                cached_coords = cached_results[location_key]
                if cached_coords: 
                    coord_dict[key] = cached_coords
            else:
                uncached_count += 1
                if uncached_count % 10 == 0 or uncached_count == 1:
                    logger.info(f"[*] hmm, geocoding uncached location {uncached_count}: {city}, {state}")
                
                coords = self.get_coordinates(city, state)
                if coords:
                    coord_dict[key] = coords
        
        logger.info(f"[*] Nice! I geocoded {uncached_count} new locations")
        
        result_df = locations_df.copy()
        
        def lookup_coords(row):
            if pd.isna(row[city_col]) or pd.isna(row[state_col]):
                return None
            key = (str(row[city_col]).strip().upper(), row[state_col])
            return coord_dict.get(key)
        
        result_df['Coordinates'] = result_df.apply(lookup_coords, axis=1)
        
        success_count = result_df['Coordinates'].notna().sum()
        logger.info(f"[*] It's lit! I geocoded {success_count}/{len(result_df)} locations ({success_count/len(result_df)*100:.1f}%)")
        
        return result_df

# load and proc data / change this to args but for now its fine

class AdvisorDataProcessor:
    def __init__(self, config=None):
        if config is None:
            config = CONFIG
        self.config = config
        self.heartland_file = config['input']['heartland_csv']
        self.locations_file = config['input']['locations_csv']
        
    def load_data(self):
        logger.info(f"[*] Reading advisor data from {self.heartland_file}")
        heartland_df = pd.read_csv(self.heartland_file, encoding='latin1')
        
        logger.info(f"[*] Reading location data from {self.locations_file}")
        locations_df = pd.read_csv(self.locations_file, encoding='latin1')
        
        return heartland_df, locations_df
        
    def preprocess_advisor_data(self, df):
        logger.info("[*] Doing some laundry...")
        
        # Convert opportunity column to numeric
        df = df.copy()
        df[' Total MF Opportunity '] = pd.to_numeric(
            df[' Total MF Opportunity '].astype(str)
                .str.replace('$', '', regex=False)
                .str.replace(',', '', regex=False), 
            errors='coerce'
        ).fillna(0.0)
        
        return df

# filter + select advisors by business rule (unique key)

    def select_relevant_advisors(self, df):
        logger.info("[*] Selecting appropriate advisors...")
        
        tier_12_mask = df['Data Driven Segment'].str.contains('1 -|2 -', na=False)
        tier_12_advisors = df[tier_12_mask].copy()
        
        tier_12_advisors['team_key'] = (
            tier_12_advisors['BR Team ID'].astype(str) + '_' + 
            tier_12_advisors['Entity ID'].astype(str) + '_' + 
            tier_12_advisors['BR Team Name'].astype(str) + '_' + 
            tier_12_advisors['Entity Name'].astype(str)
        )
        
        unique_teams = tier_12_advisors.drop_duplicates('team_key').copy()
        base_advisors_sf_ids = unique_teams['SF Contact ID'].dropna().unique()
        
        additional_advisors_sf_ids = []
        high_value_threshold = self.config['opportunity']['high_value_threshold']
        high_value_branches_ids = df[df[' Total MF Opportunity '] >= high_value_threshold]['DST Branch Internal ID'].unique()
        
        for branch_id in high_value_branches_ids:
            branch_data = df[df['DST Branch Internal ID'] == branch_id].copy()
            
            has_tier12 = branch_data['Data Driven Segment'].str.contains('1 -|2 -', na=False).any()
            
            if not has_tier12:
                branch_total = branch_data[' Total MF Opportunity '].sum()
                total_advisors = len(branch_data['SF Contact ID'].dropna().unique())
                
                if total_advisors > 0:
                    opp_per_advisor = branch_total / total_advisors
                    opp_threshold = self.config['opportunity']['opportunity_per_advisor_threshold']
                    
                    if opp_per_advisor >= opp_threshold:
                        branch_other_advisors = branch_data[
                            (branch_data['Data Driven Segment'].str.contains('3 -|4 -', na=False)) | 
                            (branch_data['Data Driven Segment'].isna())
                        ]['SF Contact ID'].dropna().unique()
                        
                        additional_advisors_sf_ids.extend(branch_other_advisors)
        
        final_advisor_ids = np.union1d(base_advisors_sf_ids, np.array(list(set(additional_advisors_sf_ids))))
        final_advisors_df = df[df['SF Contact ID'].isin(final_advisor_ids)].copy()
        
        team_name_mask = final_advisors_df['BR Team Name'].isna()
        final_advisors_df.loc[team_name_mask, 'BR Team Name'] = final_advisors_df.loc[team_name_mask, 'Entity Name']
        
        logger.info(f"[*] Selected {len(final_advisors_df)} final advisor entries")
        return final_advisors_df
    
    def combine_with_location_data(self, advisor_df, location_df, geocoding_manager):
        logger.info("[*] Combining advisor data with location information. One moment...")
        unique_states = advisor_df['Office State'].dropna().unique()
        relevant_locations = location_df[location_df['Official USPS State Code'].isin(unique_states)]
        city_coords_cache = {}
        for _, row in relevant_locations.iterrows():
            try:
                city_name = str(row['Official USPS city name']).strip().upper()
                state_code = row['Official USPS State Code']
                lat, lon = map(float, str(row['Geo Point']).split(','))
                city_coords_cache[(city_name, state_code)] = (lat, lon)
            except Exception as e:
                logger.warning(f"Error processing location row: {e}")
        conn = sqlite3.connect(geocoding_manager.cache_db)
        cursor = conn.cursor()
        for (city, state), (lat, lon) in city_coords_cache.items():
            cache_key = f"{city}_{state}"
            cursor.execute(
                "INSERT OR REPLACE INTO geocode_cache VALUES (?, ?, ?, ?)",
                (cache_key, lat, lon, int(time.time()))
            )
        conn.commit()
        conn.close()
        geocoded_df = geocoding_manager.batch_geocode(advisor_df)
        result_df = geocoded_df.dropna(subset=['Coordinates'])
        logger.info(f"[*] Final dataset contains {len(result_df)} advisors with valid coordinates")
        return result_df

# extract coordinates / vectorized operations

class ClusterAnalyzer:
    def __init__(self, config=None):
        if config is None:
            config = CONFIG['clustering']
        self.config = config
        self.n_clusters = config['n_clusters']
        self.random_state = config['random_state']
        self.colors = config['colors']
        
    def cluster_locations(self, df):
        logger.info(f"[*] Performing optimized AI clustering with {self.n_clusters} clusters")
        unique_coords_df = (
            df[['Coordinates']]
            .drop_duplicates()
            .dropna()
            .assign(lat=lambda d: d['Coordinates'].apply(lambda x: x[0]),
                    lon=lambda d: d['Coordinates'].apply(lambda x: x[1]))
        )
        
        unique_coords = unique_coords_df[['lat', 'lon']].values
        
        if len(unique_coords) < self.n_clusters:
            logger.warning(f"Not enough unique locations ({len(unique_coords)}) for {self.n_clusters} clusters. Assigning all to cluster 0.")
            df['Cluster'] = 0
            return df, None
            
        kmeans = KMeans(
            n_clusters=self.n_clusters, 
            random_state=self.random_state,
            n_init=10,
            max_iter=300,  
            tol=1e-4       
        )
        cluster_labels = kmeans.fit_predict(unique_coords)
        coord_to_cluster = {}
        for i, coord in enumerate(unique_coords):
            key = (round(coord[0], 6), round(coord[1], 6))
            coord_to_cluster[key] = int(cluster_labels[i])
        
        def get_cluster(coord):
            if pd.isna(coord):
                return np.nan
            key = (round(coord[0], 6), round(coord[1], 6))
            return coord_to_cluster.get(key, 0)
            
        df['Cluster'] = df['Coordinates'].apply(get_cluster)
        
        logger.info("[*] Clustering complete")
        return df, kmeans

class MarkerFactory:
    def __init__(self, config=None):
        if config is None:
            config = CONFIG
        self.config = config
        self.cluster_colors = config['clustering']['colors']
        
    def get_map_color_radius(self, opportunity, all_opportunities):
        if opportunity >= 100_000_000: 
            color = 'red'
        elif opportunity >= 50_000_000: 
            color = 'orange'
        elif opportunity >= 25_000_000: 
            color = 'yellow'
        else: 
            color = 'green'

        if not all_opportunities or opportunity <= 0:
            return color, 8
            
        valid_opportunities = [opp for opp in all_opportunities if opp > 0]
        if not valid_opportunities:
            return color, 8
            
        log_opp = np.log10(opportunity + 1)
        log_min = np.log10(min(valid_opportunities) + 1)
        log_max = np.log10(max(valid_opportunities) + 1)
        
        if log_max == log_min: 
            radius = 8
        else: 
            radius = 8 + 12 * (log_opp - log_min) / (log_max - log_min)
            
        return color, max(8, radius)
    
    def prepare_branch_details(self, advisors_group):
        branch_details_list = []
        
        for branch_id, branch_advisors_df in advisors_group.groupby('DST Branch Internal ID'):
            first_advisor = branch_advisors_df.iloc[0]
            
            advisors_html_list = []
            individual_advisors_data = []
            
            for _, adv_row in branch_advisors_df.iterrows():
                segment = adv_row['Data Driven Segment'] if pd.notna(adv_row['Data Driven Segment']) else 'No Segment'
                advisors_html_list.append(f"- {adv_row['BR Team Name']} ({segment})")
                
                individual_advisors_data.append({
                    'sf_contact_id': adv_row['SF Contact ID'],
                    'team_name': adv_row['BR Team Name'],
                    'segment': segment,
                    'advisor_total_mf_opportunity': adv_row[' Total MF Opportunity '] 
                })
            
            branch_opportunity = branch_advisors_df[' Total MF Opportunity '].sum()
            branch_details_list.append({
                'id': branch_id, 
                'address': first_advisor['Office Address'],
                'office_city': first_advisor['Office City'],
                'office_state': first_advisor['Office State'],
                'firm_name': first_advisor['DST Firm Name'],
                'branch_opportunity_value': branch_opportunity,
                'opportunity_str': f"${branch_opportunity/1000000:.2f}M",
                'num_advisors': len(branch_advisors_df),
                'advisors_html': "<br>".join(advisors_html_list), 
                'individual_advisors_data': individual_advisors_data 
            })
            
        return branch_details_list
    
# put it in html map

    def generate_popup_content(self, loc_data, cluster_colors_dict):
        options_html = ""
        for idx in range(self.config['clustering']['n_clusters']):
            selected = 'selected' if loc_data['current_cluster'] == idx else ''
            options_html += f'<option value="{idx}" {selected}>Cluster {idx}</option>'
            
        selected_other = 'selected' if loc_data['current_cluster'] == 'Other' else ''
        options_html += f'<option value="Other" {selected_other}>Other</option>'

        branch_html = ""
        for branch in loc_data["branch_details_for_popup"]:
            branch_html += f"""
            <b>Branch ID:</b> {branch['id']}<br>
            <b>Firm:</b> {branch.get('firm_name', 'N/A')}<br> 
            <b>Office Address:</b> {branch['address']}<br>
            <b>Office:</b> {branch['office_city']}, {branch['office_state']}<br>
            <b>Branch Opportunity:</b> {branch['opportunity_str']}<br> 
            <b>Advisors ({branch['num_advisors']}):</b><br>
            {branch['advisors_html']}<hr>
            """
            
        return f"""
        <div style="max-height: 250px; overflow-y: auto;">
        <h4>Location Total: ${(loc_data['opportunity'] / 1000000):.2f}M</h4>
        Current Cluster: <strong style="color:{cluster_colors_dict.get(loc_data['current_cluster'], 'black')}">{loc_data['current_cluster']}</strong><br>
        Original Cluster: {loc_data['initial_cluster']}<br>
        <label for="cluster_select_{loc_data['id']}">Change Cluster:</label>
        <select id="cluster_select_{loc_data['id']}" onchange="handleClusterChange('{loc_data['id']}', this.value)">
        {options_html}
        </select><hr>{branch_html}</div>
        """
    
    def create_map_data(self, clustered_df):
        logger.info("[*] Learning map data! Straight up!")
        
        map_data = []
        all_location_opportunities = []
        
        location_groups = clustered_df.groupby('Coordinates')
        
        for i, (coord_tuple, group_df) in enumerate(location_groups):
            location_total_opportunity = group_df.groupby('DST Branch Internal ID')[' Total MF Opportunity '].sum().sum()
            all_location_opportunities.append(location_total_opportunity)
            initial_cluster = int(group_df['Cluster'].iloc[0]) if not group_df['Cluster'].empty and pd.notna(group_df['Cluster'].iloc[0]) else 0
            branch_details = self.prepare_branch_details(group_df)
            map_data.append({
                'id': f"loc_{i}", 
                'lat': coord_tuple[0], 
                'lon': coord_tuple[1],
                'opportunity': location_total_opportunity,
                'initial_cluster': initial_cluster, 
                'current_cluster': initial_cluster,
                'branch_details_for_popup': branch_details
            })
        
        return map_data, all_location_opportunities

# fixed json / added kmeans ai

class MapGenerator:
    def __init__(self, config=None):
        if config is None:
            config = CONFIG
        self.config = config
        self.marker_factory = MarkerFactory(config)
        self.output_dir = config['output']['dir']
        self.map_filename = config['output']['map_filename']
        
        self.template_env = jinja2.Environment(
            loader=jinja2.FileSystemLoader('templates')
        )
        
        os.makedirs(self.output_dir, exist_ok=True)
        
        os.makedirs('templates', exist_ok=True)
        
    def create_interactive_map(self, map_data, all_opportunities, kmeans=None):
        logger.info("[*] Creating interactive map!")
        
        map_data = self._ensure_json_serializable(map_data)
        
        if map_data:
            map_center_lat = np.mean([loc['lat'] for loc in map_data])
            map_center_lon = np.mean([loc['lon'] for loc in map_data])
        else:
            map_center_lat, map_center_lon = self.config['map']['default_center']
            
        interactive_map = folium.Map(
            location=[map_center_lat, map_center_lon], 
            zoom_start=self.config['map']['default_zoom']
        )
        
        n_clusters = self.config['clustering']['n_clusters']
        cluster_colors = self.config['clustering']['colors'][:n_clusters]
        if len(cluster_colors) < n_clusters:
            cluster_colors.extend(['grey'] * (n_clusters - len(cluster_colors)))
            
        cluster_colors_dict = {i: cluster_colors[i] for i in range(n_clusters)}
        cluster_colors_dict['Other'] = 'grey'
        
        for loc_data in map_data:
            initial_marker_color = cluster_colors_dict.get(loc_data['initial_cluster'], 'grey')
            
            popup_html = self.marker_factory.generate_popup_content(loc_data, cluster_colors_dict)
            
            _, radius = self.marker_factory.get_map_color_radius(
                loc_data['opportunity'], 
                all_opportunities
            )
            marker = folium.CircleMarker(
                location=[loc_data['lat'], loc_data['lon']],
                radius=radius, 
                color=initial_marker_color, 
                fill=True, 
                fill_color=initial_marker_color,
                popup=folium.Popup(popup_html, max_width=350), 
                weight=2
            )
            marker.options.update({'customId': loc_data['id']})
            marker.add_to(interactive_map)
        
        if kmeans and hasattr(kmeans, 'cluster_centers_') and kmeans.cluster_centers_ is not None and len(kmeans.cluster_centers_) > 0:
            for i, center in enumerate(kmeans.cluster_centers_):
                folium.CircleMarker(
                    location=[center[0], center[1]], 
                    radius=10, 
                    color='black', 
                    fill=True, 
                    fill_color='dimgray',
                    popup=f'Original K-Means Center {i}', 
                    weight=2, 
                    dash_array='5, 5'
                ).add_to(interactive_map)
        
        self._create_js_template()
        js_content = self._generate_javascript(map_data, cluster_colors_dict)
        interactive_map.get_root().html.add_child(folium.Element(js_content))
        interactive_map.get_root().html.add_child(folium.Element(self._generate_legend_html()))
        interactive_map.get_root().html.add_child(folium.Element(self._generate_map_ready_script()))
        map_path = os.path.join(self.output_dir, self.map_filename)
        interactive_map.save(map_path)
        logger.info(f"[*] Interactive map saved to {map_path}")
        return map_path
    
    def _ensure_json_serializable(self, data):
        if isinstance(data, dict):
            return {k: self._ensure_json_serializable(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._ensure_json_serializable(item) for item in data]
        elif isinstance(data, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64,
                             np.uint8, np.uint16, np.uint32, np.uint64)):
            return int(data)
        elif isinstance(data, (np.float16, np.float32, np.float64)):
            return float(data)
        elif isinstance(data, np.bool_):
            return bool(data)
        elif isinstance(data, np.ndarray):
            return self._ensure_json_serializable(data.tolist())
        else:
            return data

    def _create_js_template(self):
        template_path = os.path.join('templates', 'map_script.js.jinja')
        if not os.path.exists(template_path):
            with open(template_path, 'w') as f:
                f.write("""
<script>
var mapData = {{ map_data|safe }};
var clusterColors = {{ cluster_colors|safe }};
var N_CLUSTERS = {{ n_clusters }};
var leafletMap;
var markerLayers = {};
const LOCAL_STORAGE_KEY = 'interactiveMapClusterAdjustments_v1'; 

function saveAdjustmentsToLocalStorage() {
    if (typeof localStorage !== 'undefined') {
        try {
            const dataToSave = mapData.map(loc => ({
                id: loc.id,
                current_cluster: loc.current_cluster
            }));
            localStorage.setItem(LOCAL_STORAGE_KEY, JSON.stringify(dataToSave));
            alert('Cluster adjustments saved to local storage!');
        } catch (e) {
            console.error('Error saving to local storage:', e);
            alert('Could not save adjustments. Local storage might be full or disabled.');
        }
    } else {
        alert('Local storage is not available in this browser.');
    }
}

function loadAdjustmentsFromLocalStorage() {
    let adjustmentsWereLoaded = false;
    if (typeof localStorage !== 'undefined') {
        try {
            const savedDataString = localStorage.getItem(LOCAL_STORAGE_KEY);
            if (savedDataString) {
                const savedAdjustments = JSON.parse(savedDataString);
                let adjustmentsAppliedCount = 0;
                mapData.forEach(loc => {
                    const savedLoc = savedAdjustments.find(sLoc => sLoc.id === loc.id);
                    if (savedLoc && typeof savedLoc.current_cluster !== 'undefined') { 
                        loc.current_cluster = savedLoc.current_cluster;
                        adjustmentsAppliedCount++;
                    }
                });
                if (adjustmentsAppliedCount > 0) {
                    console.log(adjustmentsAppliedCount + ' cluster adjustments loaded from local storage.');
                    adjustmentsWereLoaded = true;
                }
            }
        } catch (e) {
            console.error('Error loading from local storage:', e);
        }
    }
    return adjustmentsWereLoaded;
}

function clearSavedAdjustments() {
    if (typeof localStorage !== 'undefined') {
        try {
            localStorage.removeItem(LOCAL_STORAGE_KEY);
            alert('Saved cluster adjustments have been cleared. Reload the page to see initial K-Means clusters or re-save new ones.');
        } catch (e) {
            console.error('Error clearing local storage:', e);
            alert('Could not clear adjustments.');
        }
    } else {
        alert('Local storage is not available in this browser.');
    }
}

function addDayCalculator() { 
    var calculatorDiv = document.createElement('div');
    calculatorDiv.id = 'day-calculator';
    calculatorDiv.style.cssText = `position: fixed; top: 70px; left: 10px; background-color: white; border: 1px solid grey; padding: 8px; font-size: 11px; z-index: 1000; border-radius: 3px; box-shadow: 0 1px 3px rgba(0,0,0,0.2); max-width: 180px;`;
    document.body.appendChild(calculatorDiv);
}

function updateDayCalculator() { 
    var calculatorDiv = document.getElementById('day-calculator');
    if (!calculatorDiv) return;
    var clusterOpportunities = {};
    var totalOpportunity = 0;
    mapData.forEach(loc => {
        if (loc.current_cluster !== 'Other') {
            if (!clusterOpportunities[loc.current_cluster]) { clusterOpportunities[loc.current_cluster] = 0; }
            clusterOpportunities[loc.current_cluster] += loc.opportunity;
            totalOpportunity += loc.opportunity;
        }
    });
    var calculatorHtml = "<strong>18-Day Schedule (Excl. Other):</strong><br>";
    for (let i = 0; i < N_CLUSTERS; i++) {
        var opp = clusterOpportunities[i] || 0;
        var days = totalOpportunity > 0 ? ((opp / totalOpportunity) * 18).toFixed(1) : "0.0";
        var color = clusterColors[i] || 'black';
        calculatorHtml += `<span style="color:${color};">●</span> Cluster ${i}: ${days} days<br>`;
    }
    calculatorDiv.innerHTML = calculatorHtml;
}

function refreshAllMarkerVisuals() { 
    console.log("Refreshing all marker visuals based on current mapData.");
    mapData.forEach(locData => {
        const marker = markerLayers[locData.id];
        if (marker) {
            const newColor = clusterColors[locData.current_cluster] || 'black';
            marker.setStyle({ color: newColor, fillColor: newColor });
            marker.setPopupContent(generateJsPopupContent(locData.id)); 
        }
    });
}

function initializeMarkerLayers(mapInstance) {
    leafletMap = mapInstance;
    mapInstance.eachLayer(function(layer) {
        if (layer instanceof L.CircleMarker && layer.options && layer.options.customId) {
            markerLayers[layer.options.customId] = layer;
        }
    });

    const adjustmentsWereLoaded = loadAdjustmentsFromLocalStorage();

    if (adjustmentsWereLoaded) {
        refreshAllMarkerVisuals();
    } 

    addDayCalculator(); 
    updateLegend(); 
    updateDayCalculator();
}

function generateJsPopupContent(locationId) { 
    var locData = mapData.find(loc => loc.id === locationId);
    if (!locData) return "";
    var optionsHtml = "";
    for (var i = 0; i < N_CLUSTERS; i++) {
        optionsHtml += `<option value="${i}" ${locData.current_cluster == i ? 'selected' : ''}>Cluster ${i}</option>`;
    }
    optionsHtml += `<option value="Other" ${locData.current_cluster === 'Other' ? 'selected' : ''}>Other</option>`;
    var branchHtml = locData.branch_details_for_popup.map(b_summary => {
        return `<b>Branch ID:</b> ${b_summary.id}<br><b>Firm:</b> ${b_summary.firm_name || 'N/A'}<br><b>Office Address:</b> ${b_summary.address}<br><b>Office:</b> ${b_summary.office_city}, ${b_summary.office_state}<br><b>Branch Opportunity:</b> ${b_summary.opportunity_str}<br><b>Advisors (${b_summary.num_advisors}):</b><br>${b_summary.advisors_html}<hr>`;
    }).join('');
    return `<div style="max-height: 250px; overflow-y: auto;"><h4>Location Total: $${(locData.opportunity / 1000000).toFixed(2)}M</h4>Current Cluster: <strong style="color:${clusterColors[locData.current_cluster] || 'black'}">${locData.current_cluster}</strong><br>Original Cluster: ${locData.initial_cluster}<br><label for="cluster_select_${locationId}">Change Cluster:</label><select id="cluster_select_${locationId}" onchange="handleClusterChange('${locationId}', this.value)">${optionsHtml}</select><hr>${branchHtml}</div>`;
}

function handleClusterChange(locationId, newClusterValue) { 
    var locData = mapData.find(loc => loc.id === locationId);
    if (!locData) { console.error("Loc data not found:", locationId); return; }
    locData.current_cluster = (newClusterValue !== "Other") ? parseInt(newClusterValue) : "Other";
    var marker = markerLayers[locationId];
    if (marker) {
        var newColor = clusterColors[locData.current_cluster] || 'black';
        marker.setStyle({ color: newColor, fillColor: newColor });
        marker.setPopupContent(generateJsPopupContent(locationId)); 
    } else { console.error("Marker not found for ID:", locationId); }
    updateLegend();
    updateDayCalculator();
}

function updateLegend() { 
    var legendContentDiv = document.getElementById('cluster-legend-content');
    if (!legendContentDiv) return;
    var stats = {};
    for (var i = 0; i < N_CLUSTERS; i++) { stats[i] = { total_opp: 0, count: 0 }; }
    stats['Other'] = { total_opp: 0, count: 0 };
    mapData.forEach(loc => {
        var key = loc.current_cluster;
        if (typeof stats[key] === 'undefined') return; 
        stats[key].total_opp += loc.opportunity;
        stats[key].count += 1;
    });
    var newLegendHtml = "<p><strong>Clusters (Manually Adjustable):</strong></p>";
    for (var i = 0; i < N_CLUSTERS; i++) {
        if (stats[i].count > 0) {
            var totalOppM = (stats[i].total_opp / 1000000).toFixed(1);
            var avgOppM = stats[i].count > 0 ? (stats[i].total_opp / stats[i].count / 1000000).toFixed(1) : "0.0";
            newLegendHtml += `<p><span style="color:${clusterColors[i] || 'black'};">●</span> Cluster ${i}:<br>&nbsp;&nbsp;$${totalOppM}M Total<br>&nbsp;&nbsp;${stats[i].count} Locations<br>&nbsp;&nbsp;$${avgOppM}M Avg/Loc</p>`;
        }
    }
    var otherTotalOppM = (stats['Other'].total_opp / 1000000).toFixed(1);
    var otherAvgOppM = stats['Other'].count > 0 ? (stats['Other'].total_opp / stats['Other'].count / 1000000).toFixed(1) : "0.0";
    newLegendHtml += `<p><span style="color:${clusterColors['Other']};">●</span> Other:<br>&nbsp;&nbsp;$${otherTotalOppM}M Total<br>&nbsp;&nbsp;${stats['Other'].count} Locations<br>&nbsp;&nbsp;$${otherAvgOppM}M Avg/Loc</p>`;
    legendContentDiv.innerHTML = newLegendHtml;
}

function exportUpdatedClusters() {
    let csvContent = "data:text/csv;charset=utf-8,";
    csvContent += "DST Branch Internal ID,Office Address,Office City,Office State,DST Firm Name,SF Contact ID,BR Team Name,Data Driven Segment,Advisor Total MF Opportunity,Current Cluster\\n";
    let rowsToExport = [];
    mapData.forEach(loc => {
        loc.branch_details_for_popup.forEach(branch_summary => {
            branch_summary.individual_advisors_data.forEach(advisor => {
                rowsToExport.push({
                    branchId: branch_summary.id,
                    address: branch_summary.address,
                    city: branch_summary.office_city,
                    state: branch_summary.office_state,
                    firmName: branch_summary.firm_name,
                    sfContactId: advisor.sf_contact_id, 
                    teamName: advisor.team_name,
                    segment: advisor.segment,
                    advisorOpportunity: advisor.advisor_total_mf_opportunity,
                    cluster: loc.current_cluster 
                });
            });
        });
    });
    rowsToExport.sort((a, b) => {
        let cityComparison = a.city.localeCompare(b.city);
        if (cityComparison !== 0) return cityComparison;
        let branchComparison = String(a.branchId).localeCompare(String(b.branchId));
        if (branchComparison !== 0) return branchComparison;
        return String(a.teamName).localeCompare(String(b.teamName));
    });
    rowsToExport.forEach(dataRow => {
        let csvRowArray = [
            dataRow.branchId,
            dataRow.address,
            dataRow.city,
            dataRow.state,
            dataRow.firmName,
            dataRow.sfContactId,
            dataRow.teamName,
            dataRow.segment,
            dataRow.advisorOpportunity,
            dataRow.cluster
        ].map(field => `"${String(field === null || typeof field === 'undefined' ? '' : field).replace(/"/g, '""')}"`).join(",");
        csvContent += csvRowArray + "\\n";
    });
    var encodedUri = encodeURI(csvContent);
    var link = document.createElement("a");
    link.setAttribute("href", encodedUri);
    link.setAttribute("download", "updated_advisor_clusters_per_advisor.csv");
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
}

function onMapReady(mapInstance) {
    initializeMarkerLayers(mapInstance);
}
</script>
                """)
    
    def _generate_javascript(self, map_data, cluster_colors_dict):
        try:
            serializable_map_data = self._ensure_json_serializable(map_data)
            serializable_colors = self._ensure_json_serializable(cluster_colors_dict)
            
            template = self.template_env.get_template('map_script.js.jinja')
            return template.render(
                map_data=json.dumps(serializable_map_data),
                cluster_colors=json.dumps(serializable_colors),
                n_clusters=self.config['clustering']['n_clusters']
            )
        except jinja2.exceptions.TemplateNotFound:
            logger.warning("Template not found, using inline JavaScript")
            return f"""
            <script>
            var mapData = {json.dumps(self._ensure_json_serializable(map_data))};
            var clusterColors = {json.dumps(self._ensure_json_serializable(cluster_colors_dict))};
            var N_CLUSTERS = {self.config['clustering']['n_clusters']};
            // Additional JavaScript would follow...
            </script>
            """
        except Exception as e:
            logger.error(f"Error generating JavaScript: {e}")
            return """
            <script>
            var mapData = [];
            var clusterColors = {"0": "red", "Other": "grey"};
            var N_CLUSTERS = 1;
            alert("Error loading map data. Please check the logs.");
            </script>
            """
    
    def _generate_legend_html(self):
        return """
        <div id="cluster-legend-container" 
             style="position: fixed; bottom: 10px; right: 10px; width: 220px; max-height: calc(100vh - 20px); 
                    border:2px solid grey; z-index:9999; background-color:white; 
                    opacity:0.9; padding:10px; font-size:12px; overflow-y: auto;">
          <div id="cluster-legend-content" style="margin-bottom:10px;">
            <p><strong>Cluster Legend (Loading...)</strong></p>
          </div>
          <button onclick="exportUpdatedClusters()" 
                  style="width:100%; margin-top:5px; padding:5px; background-color:#4CAF50; color:white; border:none; cursor:pointer; font-size:12px; border-radius: 3px;">
            Export Clusters CSV
          </button>
          <button onclick="saveAdjustmentsToLocalStorage()" 
                  style="width:100%; margin-top:5px; padding:5px; background-color:#007bff; color:white; border:none; cursor:pointer; font-size:12px; border-radius: 3px;">
            Save Adjustments
          </button>
          <button onclick="clearSavedAdjustments()" 
                  style="width:100%; margin-top:5px; padding:5px; background-color:#dc3545; color:white; border:none; cursor:pointer; font-size:12px; border-radius: 3px;">
            Clear Saved
          </button>
        </div>
        """
    
    def _generate_map_ready_script(self):
        return """
        <script>
        (function() { 
            var checkMapInterval = setInterval(function() {
                var mapInstance = null;
                if (typeof L !== 'undefined' && L.DomUtil) {
                    var mapDivs = document.querySelectorAll('.folium-map');
                    if (mapDivs.length > 0) {
                        for (var i = 0; i < mapDivs.length; i++) {
                            if (mapDivs[i]._leaflet_map) { mapInstance = mapDivs[i]._leaflet_map; break; }
                        }
                        if (!mapInstance) {
                            for (var key in window) {
                                if (window.hasOwnProperty(key) && window[key] instanceof L.Map) { mapInstance = window[key]; break; }
                            }
                        }
                    }
                }
                if (mapInstance) {
                    clearInterval(checkMapInterval);
                    if (typeof onMapReady === 'function') { onMapReady(mapInstance); } 
                    else { console.error('onMapReady function not defined.'); }
                }
            }, 200);
        })();
        </script>
        """


# put js in a seperate file next time bozo 

def main():
    """Main application entry point with performance optimizations"""
    start_time = time.time()
    logger.info("[*] Starting advisor map generation")
    
    try:
        required_files = [CONFIG['input']['heartland_csv'], CONFIG['input']['locations_csv']]
        missing_files = [f for f in required_files if not os.path.exists(f)]
        if missing_files:
            raise FileNotFoundError(f"Required input files missing: {', '.join(missing_files)}")
        data_processor = AdvisorDataProcessor(CONFIG)
        geocoding_manager = GeocodingManager(CONFIG['geocoding'])
        cluster_analyzer = ClusterAnalyzer(CONFIG['clustering'])
        map_generator = MapGenerator(CONFIG)
        logger.info("[*] Step 1/6: Loading data files")
        heartland_df, locations_df = data_processor.load_data()
        clean_df = data_processor.preprocess_advisor_data(heartland_df)
        logger.info("[*] Step 2/6: Selecting relevant advisors")
        advisors_df = data_processor.select_relevant_advisors(clean_df)
        logger.info(f"[*] Processing {len(advisors_df)} advisor records")
        logger.info("[*] Step 3/6: Geocoding advisor locations")
        geocoded_df = data_processor.combine_with_location_data(advisors_df, locations_df, geocoding_manager)
        logger.info("[*] Step 4/6: Performing spatial clustering")
        clustered_df, kmeans = cluster_analyzer.cluster_locations(geocoded_df)
        logger.info("[*] Step 5/6: Preparing visualization data")
        marker_factory = MarkerFactory(CONFIG)
        map_data, all_opportunities = marker_factory.create_map_data(clustered_df)
        logger.info("[*] Step 6/6: Generating interactive map")
        map_path = map_generator.create_interactive_map(map_data, all_opportunities, kmeans)
        if 'Cluster' in clustered_df.columns:
            export_path = CONFIG['output']['advisors_export']
            logger.info(f"[*] Exporting advisor data to {export_path}")
            clustered_df.to_csv(
                export_path, 
                index=False,
                quoting=1, 
                date_format='%Y-%m-%d'
            )
            logger.info(f"[*] Saved all selected advisors with original cluster info to '{export_path}'")
        elapsed_time = time.time() - start_time
        processed_per_second = len(advisors_df) / max(0.001, elapsed_time)
        logger.info(f"[*] Process completed successfully in {elapsed_time:.2f} seconds")
        logger.info(f"[*] Processing rate: {processed_per_second:.1f} records/second")
        logger.info(f"[*] Interactive map saved to {map_path}\n")
        logger.info("[*] Please don't forget to give Sam a raise")
        return 0
    except Exception as e:
        logger.exception(f"Oops! : {str(e)}")
        return 1

if __name__ == '__main__':
    exit_code = main()
    exit(exit_code)