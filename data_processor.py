"""
SARSA Data Processor - PDF Parsing & Real-Time Data Streaming
Parses DGMS PDFs, extracts accident data, and simulates Pathway streaming
"""

import fitz  # PyMuPDF
import pandas as pd
import re
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import json
import io


# =========================================================
#                 PDF Accident Parser
# =========================================================
class PDFAccidentParser:
    """
    Parses DGMS accident reports (PDF) and extracts structured data.
    Uses PyMuPDF for text extraction and regex patterns for entity extraction.
    """

    def __init__(self):
        self.accident_pattern = re.compile(
            r'(\d{4}-\d{2}-\d{2})|'
            r'(MINE|COAL|ORE|LIMESTONE|GRANITE|MANGANESE|COPPER|ZINC|IRON)',
            re.IGNORECASE
        )

    # ----------------------------------------------------
    # Main PDF parsing entry
    # ----------------------------------------------------
    def parse_pdf(self, pdf_file) -> pd.DataFrame:
        """
        Parse DGMS PDF report and extract accident records.
        """
        try:
            # Read PDF
            if isinstance(pdf_file, str):
                doc = fitz.open(pdf_file)
            else:
                pdf_bytes = pdf_file.read()
                doc = fitz.open(stream=pdf_bytes, filetype="pdf")

            all_text = ""
            tables_data = []

            # Extract all pages' text
            for page_num in range(len(doc)):
                page = doc[page_num]
                all_text += page.get_text("text")
                tables = self._extract_tables_from_page(page)
                tables_data.extend(tables)
            doc.close()

            # Parse accident records
            accidents = self._parse_accidents_from_text(all_text)
            
            # [FIXED] Commented out the call to the undefined method
            # table_accidents = self._parse_accidents_from_tables(tables_data)
            
            # [FIXED] Removed table_accidents from this list
            all_accidents = accidents

            if not all_accidents:
                print("⚠️ No accident data found in PDF. Please check the file format.")
                return self._get_empty_dataframe()

            df = pd.DataFrame(all_accidents)
            df = self._clean_and_enrich_data(df)
            return df

        except Exception as e:
            print(f"Error parsing PDF: {e}")
            return self._get_empty_dataframe()

    # ----------------------------------------------------
    # Helpers for PDF text and table extraction
    # ----------------------------------------------------
    def _extract_tables_from_page(self, page) -> List[List]:
        """Extract table-like structures from a PDF page."""
        tables = []
        try:
            blocks = page.get_text("dict")["blocks"]
            for block in blocks:
                if "lines" in block:
                    for line in block["lines"]:
                        text_items = [span["text"] for span in line["spans"] if "text" in span]
                        if text_items:
                            tables.append(text_items)
        except:
            pass
        return tables
    
    # ----------------------------------------------------
    # [UNDEFINED METHOD - PLACEHOLDER]
    # ----------------------------------------------------
    def _parse_accidents_from_tables(self, tables_data: List) -> List[Dict]:
        """
        [NOTE] This method was called but not defined. 
        It is a placeholder. The call in parse_pdf() has been commented out.
        """
        print("Note: _parse_accidents_from_tables is not implemented.")
        return []

    # ----------------------------------------------------
    # Main text-based accident extraction  (ROBUST VERSION)
    # ----------------------------------------------------
    def _parse_accidents_from_text(self, text: str) -> List[Dict]:
        """
        Robust splitting: finds each accident entry by locating 'Date' lines (handles
        numbered lines like '12.  Date - 07/01/15') and slices the text into segments.
        Falls back to other heuristics if no 'Date' found.
        """
        accidents = []
        if not text or len(text.strip()) == 0:
            return accidents

        # Normalize PDF artifacts
        text = text.replace('\xa0', ' ')
        text = re.sub(r'-\n', '', text)  # remove hyphenated line breaks

        # Pattern for date-start lines (capture start index)
        # [FIXED] This regex now looks for the '1. Date' pattern which is the
        # reliable separator in your PDF, instead of trying to find the
        # date numbers on the same line.
        date_start_re = re.compile(
            r'(^\s*\d+\.\s*Date)',
            flags=re.MULTILINE
        )
        starts = [m.start() for m in date_start_re.finditer(text)]

        if starts:
            # add final boundary
            starts.append(len(text))
            for i in range(len(starts) - 1):
                chunk = text[starts[i]:starts[i + 1]].strip()
                if len(chunk) < 40:
                    continue
                # [MODIFIED] Using a simpler check first
                # if self._is_accident_record(chunk):
                acc = self._extract_accident_data(chunk)
                if acc:
                    accidents.append(acc)
            return accidents

        # Fallback 1: split on "Person(s) Killed"
        ps_re = re.compile(r'Person\(s\)\s+Killed', flags=re.IGNORECASE)
        if ps_re.search(text):
            chunks = re.split(r'\n\s*\d+\.\s+', text)
            for chunk in chunks:
                chunk = chunk.strip()
                if len(chunk) < 40:
                    continue
                if self._is_accident_record(chunk):
                    acc = self._extract_accident_data(chunk)
                    if acc:
                        accidents.append(acc)
            return accidents

        # Fallback 2: naive split by double newlines
        for para in re.split(r'\n{2,}', text):
            para = para.strip()
            if len(para) < 40:
                continue
            if self._is_accident_record(para):
                acc = self._extract_accident_data(para)
                if acc:
                    accidents.append(acc)

        return accidents

    def _is_accident_record(self, text: str) -> bool:
        """Detect accident-related text."""
        keywords = ['fatal', 'death', 'killed', 'accident', 'incident',
                    'injury', 'collapse', 'fall', 'explosion', 'electrocution']
        return any(k in text.lower() for k in keywords)

    # ----------------------------------------------------
    # Field-level extraction using regex and heuristics
    # ----------------------------------------------------
    def _extract_accident_data(self, text: str) -> Optional[Dict]:
        """Extract structured accident data from text block."""
        try:
            accident = {}

            # --- Date ---
            for pattern in [r'(\d{4}-\d{2}-\d{2})',
                            r'(\d{2}/\d{2}/\d{4})',
                            r'(\d{2}\.\d{2}\.\d{4})',
                            r'(\d{2}/\d{2}/\d{2})'  # [FIXED] Added dd/mm/yy format
                           ]:
                m = re.search(pattern, text)
                if m:
                    date_str = m.group(1)
                    try:
                        if '-' in date_str:
                            accident['date'] = date_str
                        elif '/' in date_str:
                            # [FIXED] Handle both 2-digit and 4-digit years
                            if len(date_str) == 8: # dd/mm/yy
                                accident['date'] = datetime.strptime(date_str, '%d/%m/%y').strftime('%Y-%m-%d')
                            else: # dd/mm/YYYY
                                accident['date'] = datetime.strptime(date_str, '%d/%m/%Y').strftime('%Y-%m-%d')
                        elif '.' in date_str:
                            accident['date'] = datetime.strptime(date_str, '%d.%m.%Y').strftime('%Y-%m-%d')
                    except:
                        pass
                    if 'date' in accident:
                        break

            # --- Mine name ---
            mine_pattern = r'(?:Mine|Colliery|Quarry)\s*[-:]?\s*\n?\s*([A-Z\s&\-\.]+?)(?:\n|,|Owner|Dist)'
            m = re.search(mine_pattern, text, re.IGNORECASE)
            if m:
                accident['mine_name'] = m.group(1).strip()
            else:
                 # Fallback for mine name (e.g., )
                 m = re.search(r'Mine\s+([A-Z\s&\-\.]+?)\n', text)
                 if m:
                     accident['mine_name'] = m.group(1).strip()


            # --- State ---
            states = [
                'Jharkhand', 'Odisha', 'Chhattisgarh', 'Maharashtra', 'Rajasthan',
                'West Bengal', 'Madhya Pradesh', 'Andhra Pradesh', 'Telangana',
                'Karnataka', 'Tamil Nadu', 'Gujarat', 'Uttar Pradesh', 'Orissa',
                'Tamilnadu', 'Assam', 'Tripura' # [FIXED] Added states from PDF
            ]
            for s in states:
                if s.lower() in text.lower():
                    accident['state'] = s
                    break

            # --- District ---
            m = re.search(r'(?:Dist\.?|District)\s*[-:]?\s*\n?\s*([A-Za-z\s]+?)(?:,|State|\n)', text, re.IGNORECASE)
            if m:
                accident['district'] = m.group(1).strip()

            # --- Owner ---
            # [FIXED] Added \n?\s* to handle names on a new line
            m = re.search(r'(?:Owner|Owner\s*-|Operator)\s*[-:]?\s*\n?\s*([A-Z\s&\-\.\(\)]+?)(?:,|\n|$)', text, re.IGNORECASE)
            if m:
                accident['owner'] = m.group(1).strip()

            # --- Victim name ---
            for pattern in [
                # [FIXED] Improved regex to find names like '1. Vijendra Singh, Driller...'
                r'Person\(s\)\s+Killed:?\s*\n?\s*(?:\d\.\s*)?([A-Za-z\s\.]+?),\s*([A-Za-z\s]+?),\s*(Male|Female),\s*(\d+)\s+Years',
                r'(?:victim|deceased|killed|person)\s*[:\-]?\s*([A-Za-z\s\.]+?)(?:\n|,|aged|age)',
                r'([A-Z][a-z]+\s+[A-Z][a-z]+)\s+\(.*?\)\s+(?:was|died|killed)'
            ]:
                m = re.search(pattern, text, re.IGNORECASE)
                if m and 'killed_person' not in accident:
                    accident['killed_person'] = m.group(1).strip()
                    if m.lastindex >= 2:
                        accident['role'] = m.group(2).strip()
                    if m.lastindex >= 4:
                        try:
                            accident['age'] = int(m.group(4))
                        except:
                            pass
                    break

            # --- Age ---
            if 'age' not in accident:
                m = re.search(r'(?:age|aged)\s*[:\-]?\s*(\d{2})', text, re.IGNORECASE)
                if m:
                    accident['age'] = int(m.group(1))

            # --- Role ---
            if 'role' not in accident:
                for r in ['driller', 'operator', 'worker', 'miner', 'supervisor',
                          'mate', 'mazdoor', 'helper', 'sirdar', 'electrician', 'blaster',
                          'labour', 'bellman', 'signalman', 'driver', 'sampler', 'scrapper'
                         ]:
                    if r in text.lower():
                        accident['role'] = r.title()
                        break

            # --- Cause category ---
            cause_map = {
                'Fall of Roof': ['roof fall', 'roof collapse', 'roof'],
                'Fall of Sides': ['side fall', 'highwall', 'bench collapse', 'slope', 'side of the bench slided', 'sides of the bench been secured'],
                'Dumpers': ['dumper', 'tipper', 'truck'],
                'Explosives': ['explosion', 'blast', 'explosive'],
                'Electricity': ['electrocution', 'electric shock', 'electrical', 'live jointed cable'],
                'Fall of Person': ['fell', 'fall from height', 'falling', 'slipped and fell'],
                'Conveyors': ['conveyor', 'belt'],
                'Machinery': ['machinery', 'equipment', 'excavator', 'tractor compressor'],
                'Winding': ['winding', 'cage', 'skip'],
                'Wagon Movements': ['wagon', 'railway'],
                'Wheeled Trackless': ['truck', 'tanker', 'tractor', 'light vehicle'],
                'Fall of Objects': ['fall of objects', 'boulder', 'stone measuring'],
                'Flying Pieces': ['flying pieces', 'rebounding rock'],
                'Drowning': ['drowning', 'fell down into water'],
                'Gas/Fire': ['blow out', 'fire', 'gas']
            }
            for cat, keys in cause_map.items():
                if any(k in text.lower() for k in keys):
                    accident['cause_category'] = cat
                    break

            # --- Description ---
            # Try to find the narrative description
            desc_m = re.search(r'(?:Male|Female),\s*\d+\s+Years\n(.*?)(?=Had\s|this accident could)', text, re.DOTALL | re.IGNORECASE)
            if desc_m:
                 accident['description'] = desc_m.group(1).strip().replace('\n', ' ')
            else:
                # Fallback to just using the chunk
                accident['description'] = text[:500].replace('\n', ' ')
            
            accident['id'] = hash(text[:100]) % 10000
            
            # Count deaths
            deaths = len(re.findall(r'\d+\.\s+[A-Za-z\s]+?,', text))
            accident['deaths'] = max(1, deaths)


            # [FIXED] Changed this to be less strict. A record is valid if it has a date.
            if 'date' in accident:
                return accident
            return None

        except Exception as e:
            print(f"Error extracting accident data: {e} | TEXT: {text[:100]}...")
            return None

    # ----------------------------------------------------
    # Cleaning and enrichment
    # ----------------------------------------------------
    def _clean_and_enrich_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure schema consistency, add derived fields."""
        required_cols = [
            'id', 'date', 'mine_name', 'owner', 'district', 'state',
            'killed_person', 'age', 'role', 'deaths',
            'cause_category', 'description'
        ]
        for c in required_cols:
            if c not in df.columns:
                df[c] = None if c != 'deaths' else 1

        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df['year'] = df['date'].dt.year

        coords = df.apply(self._get_approximate_coordinates, axis=1)
        df['lat'] = coords.apply(lambda x: x[0])
        df['lon'] = coords.apply(lambda x: x[1])

        text_cols = ['mine_name', 'owner', 'district', 'state', 'cause_category', 'description']
        for col in text_cols:
            if col not in df.columns:
                df[col] = ''
        df['raw_text_for_search'] = df[text_cols].fillna('').agg(' '.join, axis=1).str.lower()
        df['mine_name_lower'] = df['mine_name'].astype(str).str.lower()
        
        # Fill missing categories based on description
        if 'cause_category' in df.columns:
             df['cause_category'] = df.apply(
                 lambda row: 'Fall of Person' if 'fall of person' in str(row['description']).lower() and pd.isna(row['cause_category']) else row['cause_category'],
                 axis=1
             )
             df['cause_category'] = df.apply(
                 lambda row: 'Fall of Sides' if 'fall of sides' in str(row['description']).lower() and pd.isna(row['cause_category']) else row['cause_category'],
                 axis=1
             )

        return df

    def _get_approximate_coordinates(self, row) -> Tuple[float, float]:
        """Approximate coordinates based on state."""
        state_coords = {
            'Jharkhand': (23.6102, 85.2799),
            'Odisha': (20.9517, 85.0985),
            'Orissa': (20.9517, 85.0985), # Alias
            'Chhattisgarh': (21.2787, 81.8661),
            'Maharashtra': (19.7515, 75.7139),
            'Rajasthan': (27.0238, 74.2179),
            'West Bengal': (22.9868, 87.8550),
            'Madhya Pradesh': (22.9734, 78.6569),
            'Andhra Pradesh': (15.9129, 79.7400),
            'Telangana': (18.1124, 79.0193),
            'Karnataka': (15.3173, 75.7139),
            'Tamil Nadu': (11.1271, 78.6569),
            'Tamilnadu': (11.1271, 78.6569), # Alias
            'Gujarat': (22.2587, 71.1924),
            'Uttar Pradesh': (26.8467, 80.9462),
            'Assam': (26.2006, 92.9376),
            'Tripura': (23.9408, 91.9882)
        }
        return state_coords.get(row.get('state', ''), (20.5937, 78.9629))

    def _get_empty_dataframe(self) -> pd.DataFrame:
        """Return empty dataframe with schema."""
        return pd.DataFrame(columns=[
            'id', 'date', 'mine_name', 'owner', 'district', 'state',
            'killed_person', 'age', 'role', 'deaths',
            'cause_category', 'lat', 'lon',
            'description', 'year', 'raw_text_for_search', 'mine_name_lower'
        ])


# =========================================================
#               Pathway Stream Simulator
# =========================================================
class PathwayStreamSimulator:
    """
    Simulates Pathway's real-time streaming capabilities.
    In production, this would connect to live data sources.
    """

    def __init__(self):
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.vector_dim = 384
        self.index = faiss.IndexFlatL2(self.vector_dim)
        self.documents = []
        self.knowledge_graph = {'entities': {}, 'relationships': []}
        self.news_alerts = []

    def ingest_data(self, df: pd.DataFrame):
        """Ingest and process accident data for RAG"""
        if df is None or len(df) == 0:
            print("Simulator received no data to ingest.")
            return
        self.df = df
        self._build_vector_store()
        self._build_knowledge_graph()

    def _build_vector_store(self):
        """Build FAISS vector store for semantic search"""
        if len(self.df) == 0:
            return
        texts = self.df['description'].fillna('').tolist()
        embeddings = self.embedding_model.encode(texts)
        self.index = faiss.IndexFlatL2(self.vector_dim)
        self.index.add(np.array(embeddings).astype('float32'))
        self.documents = texts
        print(f"Vector store built with {len(self.documents)} documents.")

    def _build_knowledge_graph(self):
        """Build knowledge graph from accident data"""
        if len(self.df) == 0:
             return
        entities = {
            'mines': self.df['mine_name'].dropna().unique().tolist(),
            'states': self.df['state'].dropna().unique().tolist(),
            'causes': self.df['cause_category'].dropna().unique().tolist(),
        }
        self.knowledge_graph['entities'] = entities

    def semantic_search(self, query: str, top_k: int = 5) -> List[int]:
        """Perform semantic search using embeddings"""
        if len(self.documents) == 0:
            return []
        query_embedding = self.embedding_model.encode([query])
        distances, indices = self.index.search(np.array(query_embedding).astype('float32'), top_k)
        return indices[0].tolist()

    def generate_live_alerts(self) -> List[Dict]:
        """Generate simulated real-time alerts"""
        from datetime import timedelta
        import random

        alerts = []
        if hasattr(self, 'df') and len(self.df) > 0:
            recent_accidents = self.df.nlargest(3, 'date') if 'date' in self.df.columns else self.df.head(3)
            for _, row in recent_accidents.iterrows():
                alert_type = random.choice(['warning', 'info'])
                cause = row.get('cause_category', 'Unknown')
                state = row.get('state', 'Unknown')
                district = row.get('district', 'Unknown')
                alerts.append({
                    'type': alert_type,
                    'time': (datetime.now() - timedelta(minutes=random.randint(5, 120))).strftime('%H:%M:%S'),
                    'message': f"New '{cause}' incident detected in {district}, {state}"
                })
        return alerts


# =========================================================
#               Regulatory Knowledge Base
# =========================================================
class RegulatoryKnowledgeBase:
    """Stores DGMS regulatory information for compliance checking."""

    def __init__(self):
        self.regulations = self._load_regulations()

    def _load_regulations(self) -> Dict:
        return {
            'MMR 106(1)': {
                'title': 'Support of Roof and Sides',
                'description': 'Adequate support must be provided to prevent fall of roof in underground mines.',
                'applies_to': ['Fall of Roof']
            },
            'MMR 106(3)': {
                'title': 'Benching and Sloping',
                'description': 'Sides of excavations must be adequately benched, sloped and secured.',
                'applies_to': ['Fall of Sides']
            },
            'MMR 118(4)': {
                'title': 'Safety Belts',
                'description': 'No person shall work at a height of more than 1.8m without safety belt.',
                'applies_to': ['Fall of Person']
            },
            'MMR 181': {
                'title': 'Safe Operation of Vehicles',
                'description': 'Vehicles must have effective brakes, proper signaling, and trained operators.',
                'applies_to': ['Dumpers', 'Transportation Machinery', 'Wheeled Trackless']
            },
            'MMR 93': {
                'title': 'Explosives Handling',
                'description': 'Proper procedures for handling, storage, and use of explosives.',
                'applies_to': ['Explosives']
            },
            'MMR 127': {
                'title': 'Electrical Safety',
                'description': 'Electrical installations must meet safety standards and be regularly inspected.',
                'applies_to': ['Electricity']
            },
            'MMR 145': {
                'title': 'Machinery Safety',
                'description': 'All machinery must have proper guards and lockout-tagout procedures.',
                'applies_to': ['Conveyors', 'Machinery']
            }
        }

    def get_violated_regulation(self, cause_category: str) -> Optional[str]:
        if not cause_category:
            return None
        for reg_id, reg_data in self.regulations.items():
            if cause_category in reg_data['applies_to']:
                return reg_id
        return None

    def get_regulation_details(self, reg_id: str) -> Optional[Dict]:
        return self.regulations.get(reg_id)


# =Example usage (you can run this part to test it)
if __name__ == "__main__":
    
    # --- 1. Test the PDF Parser ---
    print("--- 1. Testing PDF Parser ---")
    parser = PDFAccidentParser()
    
    # Replace with the actual path to your PDF
    pdf_path = "VOLUME_II_NON_COAL_2015.pdf" 
    
    try:
        accident_df = parser.parse_pdf(pdf_path)
        
        if not accident_df.empty:
            print(f"✅ Successfully parsed {len(accident_df)} accident records.")
            print("--- Sample Data (First 5 Rows) ---")
            print(accident_df.head().to_string())
            print("\n--- Columns ---")
            print(accident_df.info())
            
            # --- 2. Test the Simulator ---
            print("\n--- 2. Testing Pathway Simulator ---")
            simulator = PathwayStreamSimulator()
            simulator.ingest_data(accident_df)
            
            # Test semantic search
            print("\n--- Semantic Search Test ---")
            search_query = "accident involving a dumper"
            results_indices = simulator.semantic_search(search_query, top_k=2)
            print(f"Search results for: '{search_query}'")
            if results_indices:
                print(simulator.df.iloc[results_indices][['date', 'description', 'cause_category']])
            else:
                print("No search results found.")
            
            # Test alerts
            print("\n--- Live Alerts Test ---")
            alerts = simulator.generate_live_alerts()
            for alert in alerts:
                print(f"ALERT [{alert['time']}]: {alert['message']} (Type: {alert['type']})")

            # --- 3. Test the Knowledge Base ---
            print("\n--- 3. Testing Regulatory Knowledge Base ---")
            kb = RegulatoryKnowledgeBase()
            sample_cause = accident_df.iloc[0]['cause_category']
            print(f"Checking regulations for cause: '{sample_cause}'")
            
            reg_id = kb.get_violated_regulation(sample_cause)
            if reg_id:
                details = kb.get_regulation_details(reg_id)
                print(f"Found Violated Regulation: {reg_id} - {details['title']}")
                print(f"Description: {details['description']}")
            else:
                print("No specific regulation found in KB for this cause.")
                
        else:
            print("Script ran, but no records were extracted. Check PDF content and regex.")

    except FileNotFoundError:
        print(f"Error: The file '{pdf_path}' was not found.")
        print("Please make sure the PDF file is in the same directory as the script,")
        print("or provide the full absolute path to the file.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
