import os
import json
from PIL import Image, UnidentifiedImageError, ImageFilter
import imagehash
from collections import defaultdict
import numpy as np
import faiss

# Optional Deep Learning imports
try:
    import torch
    import torchvision.models as models
    import torchvision.transforms as transforms
    from scipy.spatial import distance
    DEEP_LEARNING_AVAILABLE = True
except ImportError:
    DEEP_LEARNING_AVAILABLE = False

DB_FILENAME = ".image_db_v2.json"
SUPPORTED_FORMATS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff'}

# --- Screenshot Heuristics ---
SCREENSHOT_KEYWORDS = ['screenshot', 'screen shot', 'captura']
EDGE_DENSITY_THRESHOLD = 0.05
COLOR_SIMPLICITY_THRESHOLD = 0.5 
TOP_COLOR_COUNT = 10
COMMON_ASPECT_RATIOS = {
    (16, 9), (9, 16), (4, 3), (3, 4), (16, 10), (10, 16), (5, 4), (4, 5)
}

# --- Deep Learning Model Setup ---
MODEL = None
TRANSFORMS = None
DEVICE = "cpu"

def init_model(status_callback):
    global MODEL, TRANSFORMS
    if not DEEP_LEARNING_AVAILABLE:
        status_callback.emit("Deep Learning libraries (PyTorch) not installed.")
        return False
    if MODEL is None:
        try:
            status_callback.emit(f"Loading Deep Learning model (ResNet50) on {DEVICE.upper()}...")
            MODEL = models.resnet50(weights=models.ResNet50_Weights.DEFAULT).to(DEVICE)
            MODEL.eval()
            MODEL = torch.nn.Sequential(*(list(MODEL.children())[:-1]))
            TRANSFORMS = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            status_callback.emit("Model loaded successfully.")
            return True
        except Exception as e:
            status_callback.emit("Error: Could not load Deep Learning model.")
            print(f"Model loading error: {e}")
            return False
    return True

def get_deep_features(image_path):
    try:
        with Image.open(image_path) as img:
            img_rgb = img.convert('RGB')
            img_t = TRANSFORMS(img_rgb)
            batch_t = torch.unsqueeze(img_t, 0).to(DEVICE)
            with torch.no_grad():
                features = MODEL(batch_t)
            return features.squeeze().cpu().numpy().tolist()
    except Exception as e:
        print(f"ERROR processing {os.path.basename(image_path)} with Deep Learning: {e}")
        return None

def get_perceptual_hash(image_path, method_key):
    try:
        with Image.open(image_path) as img:
            img.load()
            img_rgb = img.convert('RGB')
            hash_func = getattr(imagehash, method_key)
            return hash_func(img_rgb)
    except (UnidentifiedImageError, OSError) as e:
        return None
    except Exception as e:
        return None

def load_database(folder_path):
    db_path = os.path.join(folder_path, DB_FILENAME)
    if os.path.exists(db_path):
        try:
            with open(db_path, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError): pass
    return {}

def save_database(folder_path, db_data):
    db_path = os.path.join(folder_path, DB_FILENAME)
    with open(db_path, 'w') as f:
        json.dump(db_data, f)

def find_screenshots(folder_path, progress_callback, status_callback):
    status_callback.emit("Finding image files...")
    image_files = [os.path.join(r, f) for r, _, fs in os.walk(folder_path) for f in fs if os.path.splitext(f)[1].lower() in SUPPORTED_FORMATS]
    total_files = len(image_files)
    if total_files == 0: return []

    screenshots_found = []
    for i, fpath in enumerate(image_files):
        progress_callback.emit(int(((i + 1) / total_files) * 100))
        status_callback.emit(f"Checking file {i+1}/{total_files}...")

        try:
            if any(keyword in os.path.basename(fpath).lower() for keyword in SCREENSHOT_KEYWORDS):
                screenshots_found.append(fpath)
                continue

            with Image.open(fpath) as img:
                score = 0
                
                w, h = img.size
                if h > 0 and w > 0:
                    gcd = np.gcd(w, h)
                    aspect_ratio = (w // gcd, h // gcd)
                    if aspect_ratio in COMMON_ASPECT_RATIOS:
                        score += 1

                small_img = img.resize((100, 100))
                colors = small_img.getcolors(100*100)
                if colors:
                    colors.sort(key=lambda t: t[0], reverse=True)
                    top_pixel_count = sum(t[0] for t in colors[:TOP_COLOR_COUNT])
                    color_simplicity_ratio = top_pixel_count / (small_img.width * small_img.height)
                    if color_simplicity_ratio >= COLOR_SIMPLICITY_THRESHOLD:
                        score += 2

                grayscale_img = img.convert('L')
                edges = grayscale_img.filter(ImageFilter.FIND_EDGES)
                edge_pixels = np.array(edges)
                edge_ratio = np.count_nonzero(edge_pixels) / edge_pixels.size
                if edge_ratio > EDGE_DENSITY_THRESHOLD:
                    score += 2

                if score >= 4:
                    screenshots_found.append(fpath)

        except (UnidentifiedImageError, OSError):
            continue

    status_callback.emit(f"Scan complete. Found {len(screenshots_found)} potential screenshots.")
    return screenshots_found

def scan_for_duplicates(folder_path, method_str, threshold, progress_callback, status_callback):
    method_key_map = {
        "Perceptual Hash (pHash)": "phash",
        "Difference Hash (dHash)": "dhash",
        "Wavelet Hash (wHash)": "whash",
        "Deep Learning (ResNet)": "resnet_features"
    }
    method_key = method_key_map[method_str]
    is_deep_learning = "Deep Learning" in method_str

    if is_deep_learning:
        if not init_model(status_callback):
            return []

    status_callback.emit("Finding image files...")
    image_files = [os.path.join(r, f) for r, _, fs in os.walk(folder_path) for f in fs if os.path.splitext(f)[1].lower() in SUPPORTED_FORMATS]
    total_files = len(image_files)
    if total_files == 0: return []

    db_data = load_database(folder_path)
    signatures = {}
    needs_saving = False
    failed_files = 0

    for i, fpath in enumerate(image_files):
        progress_callback.emit(int(((i + 1) / total_files) * 50))
        status_callback.emit(f"Analyzing file {i+1}/{total_files}...")
        
        db_data.setdefault(fpath, {})

        if db_data[fpath].get(method_key) is None:
            needs_saving = True
            if is_deep_learning:
                signature = get_deep_features(fpath)
            else:
                h = get_perceptual_hash(fpath, method_key)
                signature = str(h) if h else None
            db_data[fpath][method_key] = signature
        
        final_signature = db_data[fpath].get(method_key)
        if final_signature:
            signatures[fpath] = imagehash.hex_to_hash(final_signature) if not is_deep_learning else final_signature
        else:
            failed_files += 1

    if needs_saving:
        status_callback.emit("Saving updated signature database...")
        save_database(folder_path, db_data)

    status_callback.emit("Comparing signatures...")
    matches = defaultdict(list)
    sig_items = list(signatures.items())
    
    if not sig_items:
        final_message = "Scan complete. No matches found."
        if failed_files > 0:
            final_message += f" ({failed_files} files could not be processed.)"
        status_callback.emit(final_message)
        return []

    if is_deep_learning:
        # Prepare data for FAISS
        # Filter out None signatures and collect paths
        valid_signatures = []
        valid_paths = []
        for p, s in sig_items:
            if s is not None:
                valid_signatures.append(s)
                valid_paths.append(p)

        if not valid_signatures:
            final_message = "Scan complete. No valid deep learning signatures found."
            if failed_files > 0:
                final_message += f" ({failed_files} files could not be processed.)"
            status_callback.emit(final_message)
            return []

        feature_vectors = np.array(valid_signatures).astype('float32')
        
        # Normalize vectors for cosine similarity with L2 index
        faiss.normalize_L2(feature_vectors)

        dimension = feature_vectors.shape[1]
        index = faiss.IndexFlatIP(dimension) # Inner Product for cosine similarity
        index.add(feature_vectors)

        # Query the index for each vector
        # We search for k+1 neighbors because the first neighbor will be the query itself
        k = min(100, len(valid_signatures)) # Search for up to 100 nearest neighbors
        D, I = index.search(feature_vectors, k + 1) # D: distances, I: indices

        # Process results to find duplicates
        # Use a Disjoint Set Union (DSU) data structure to manage groups efficiently
        parent = {i: i for i in range(len(valid_paths))}
        
        def find(i):
            if parent[i] == i:
                return i
            parent[i] = find(parent[i])
            return parent[i]

        def union(i, j):
            root_i = find(i)
            root_j = find(j)
            if root_i != root_j:
                parent[root_j] = root_i
                return True
            return False

        for i in range(len(valid_paths)):
            path1 = valid_paths[i]
            progress_callback.emit(50 + int(((i + 1) / len(valid_paths)) * 50))
            
            # Iterate through neighbors (excluding self)
            for j_idx in range(1, D.shape[1]): # Start from 1 to skip self-match
                neighbor_index = I[i, j_idx]
                similarity = D[i, j_idx] # Inner product is cosine similarity for L2 normalized vectors
                
                if (similarity * 100) >= threshold:
                    path2 = valid_paths[neighbor_index]
                    union(i, neighbor_index)
                else:
                    # Since results are sorted by similarity, we can break early
                    break
        
        # Form duplicate groups from DSU structure
        groups_map = defaultdict(list)
        for i, path in enumerate(valid_paths):
            root = find(i)
            groups_map[root].append(path)
        
        duplicate_groups = [group for group in groups_map.values() if len(group) > 1]

    else: # Perceptual Hashing (pHash, dHash, wHash)
        # Implement a simple bucketing strategy for perceptual hashes
        # Group hashes by a prefix to reduce comparisons
        # This is a basic LSH-like approach for Hamming distance
        
        # Determine bucket size (e.g., first 8 bits of the hash)
        # Assuming hashes are 64-bit (imagehash default)
        BUCKET_BITS = 8 
        buckets = defaultdict(list)

        for path, sig in sig_items:
            if sig is not None:
                # Convert hash to binary string and take prefix
                # imagehash returns a Hex hash, convert to binary for bucketing
                binary_sig = bin(int(str(sig), 16))[2:].zfill(64) # Ensure 64 bits
                bucket_key = binary_sig[:BUCKET_BITS]
                buckets[bucket_key].append((path, sig))

        # Use DSU for grouping
        parent = {path: path for path, _ in sig_items if _ is not None}
        
        def find(p):
            if parent[p] == p:
                return p
            parent[p] = find(parent[p])
            return parent[p]

        def union(p1, p2):
            root_p1 = find(p1)
            root_p2 = find(p2)
            if root_p1 != root_p2:
                parent[root_p2] = root_p1
                return True
            return False

        processed_paths = set() # To avoid redundant comparisons and processing

        for i, (path1, sig1) in enumerate(sig_items):
            if sig1 is None or path1 in processed_paths:
                continue

            progress_callback.emit(50 + int(((i + 1) / len(sig_items)) * 50))
            
            binary_sig1 = bin(int(str(sig1), 16))[2:].zfill(64)
            bucket_key1 = binary_sig1[:BUCKET_BITS]

            # Compare within the same bucket
            if bucket_key1 in buckets:
                for path2, sig2 in buckets[bucket_key1]:
                    if sig2 is None or path1 == path2 or path2 in processed_paths:
                        continue
                    
                    if sig1 - sig2 <= threshold:
                        union(path1, path2)
            
            # Optionally, compare with adjacent buckets if threshold is large enough
            # This is more complex and might not be necessary for small thresholds
            # For now, we stick to same-bucket comparisons for simplicity and efficiency.
            
            processed_paths.add(path1)

        # Form duplicate groups from DSU structure
        groups_map = defaultdict(list)
        for path, _ in sig_items:
            if path in parent: # Only consider paths that were processed
                root = find(path)
                groups_map[root].append(path)
        
        duplicate_groups = [group for group in groups_map.values() if len(group) > 1]
    
    final_message = f"Scan complete. Found {len(duplicate_groups)} duplicate groups."
    if failed_files > 0:
        final_message += f" ({failed_files} files could not be processed.)"
    status_callback.emit(final_message)
    return duplicate_groups

    duplicate_groups = [group for group in matches.values() if len(group) > 1]
    final_message = f"Scan complete. Found {len(duplicate_groups)} duplicate groups."
    if failed_files > 0:
        final_message += f" ({failed_files} files could not be processed.)"
    status_callback.emit(final_message)
    return duplicate_groups