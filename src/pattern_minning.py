import pandas as pd
import ast
from pathlib import Path
from prefixspan import PrefixSpan
import re
 
# ---- Inputs ----
PER_PATIENT = Path('data/modified_per_patient.csv') #Change the location
OUT_TXT = Path("data/prefixspan_ccsr_patterns_clean.txt")#Change the location

# ---- Load per-patient sequences ----
df = pd.read_csv(PER_PATIENT)
 
def is_valid_disease(s):
    """Check if the disease entry is valid (not NaN, empty, or just whitespace)."""
    if pd.isna(s):
        return False
    if not str(s).strip():
        return False
    if str(s).strip() in ['', ' ', 'nan', 'NaN', 'NULL', 'null']:
        return False
    return True
 
def clean_code(code):
    """Clean individual code by removing extra quotes and whitespace."""
    if not isinstance(code, str):
        code = str(code)
 
    # Remove leading/trailing whitespace
    code = code.strip()
 
    # Remove outer quotes if present (both single and double)
    # Handle cases like "'XXX000'" -> "XXX000"
    if len(code) >= 2:
        if (code.startswith("'") and code.endswith("'")) or \
           (code.startswith('"') and code.endswith('"')):
            code = code[1:-1]
 
    # Remove any remaining whitespace
    code = code.strip()
 
    return code
 
def is_valid_code(code):
    """Check if a code is valid after cleaning."""
    if not code:
        return False
    if not code.strip():
        return False
    if code.strip() in ['', ' ', 'nan', 'NaN', 'NULL', 'null']:
        return False
 
    # Add more invalid patterns here as needed
    invalid_codes = {
        'XXX000', 'XXX111',  # Already identified as unwanted
        'END005', 'FAC025',  # Based on your output, these seem invalid too
        # Add more codes here if needed
    }
 
    # Check if it's in the invalid codes set
    if code in invalid_codes:
        return False
 
    # Check for patterns that look like empty or whitespace
    if re.match(r'^[\s\'\"]*$', code):
        return False
 
    return True
 
def parse_list(s):
    """Parse string into list, remove empty or whitespace-only items."""
    if not is_valid_disease(s):
        return []
    try:
        lst = ast.literal_eval(str(s))
        if not isinstance(lst, list):
            lst = [lst]
    except Exception:
        lst = [t for t in str(s).split(";") if t]
 
    # Clean each code and filter out invalid ones
    cleaned_codes = []
    for item in lst:
        if isinstance(item, str):
            cleaned_code = clean_code(item)
            if is_valid_code(cleaned_code):
                cleaned_codes.append(cleaned_code)
 
    return cleaned_codes
 
def clean_sequences(sequences):
    """Additional cleaning step to ensure no invalid codes slip through."""
    cleaned_sequences = []
 
    for seq in sequences:
        cleaned_seq = []
        for code in seq:
            cleaned_code = clean_code(str(code))
            if is_valid_code(cleaned_code):
                cleaned_seq.append(cleaned_code)
 
        # Only keep sequences that have at least one valid code
        if cleaned_seq:
            cleaned_sequences.append(cleaned_seq)
 
    return cleaned_sequences
 
# ---- Filter out invalid diseases first ----
print(f"Original dataframe shape: {df.shape}")
 
# Determine which column to use
if "disease_codes" in df.columns:
    ccsr_column = "disease_codes"
elif "patient_ccsr_codes_joined" in df.columns:
    ccsr_column = "patient_ccsr_codes_joined"
else:
    raise ValueError("Expected 'patient_ccsr_codes' or 'patient_ccsr_codes_joined' in CSV.")
 
# Filter out rows with invalid diseases
df_filtered = df[df[ccsr_column].apply(is_valid_disease)].copy()
print(f"After filtering invalid diseases: {df_filtered.shape}")
 
# Parse sequences from filtered data
sequences = df_filtered[ccsr_column].apply(parse_list).tolist()
 
# Remove empty sequences entirely
sequences = [seq for seq in sequences if seq]
print(f"Number of valid sequences before final cleaning: {len(sequences)}")
 
# ---- Clean all sequences thoroughly ----
sequences_cleaned = clean_sequences(sequences)
print(f"Number of valid sequences after cleaning: {len(sequences_cleaned)}")
 
# Count statistics
total_codes_before = sum(len(seq) for seq in sequences)
total_codes_after = sum(len(seq) for seq in sequences_cleaned)
print(f"Total codes before cleaning: {total_codes_before}")
print(f"Total codes after cleaning: {total_codes_after}")
print(f"Codes removed: {total_codes_before - total_codes_after}")
 
# Show some example cleaned sequences
print("\nExample cleaned sequences (first 5):")
for i, seq in enumerate(sequences_cleaned[:5]):
    print(f"Sequence {i+1}: {seq}")
 
# ---- Run PrefixSpan on cleaned data ----
min_support_subjects = 1000  # adjust based on dataset size
ps = PrefixSpan(sequences_cleaned)
patterns = ps.frequent(min_support_subjects)
 
# Sort by support desc, then by pattern length desc
patterns.sort(key=lambda x: (x[0], len(x[1])), reverse=True)
 
# ---- Write TXT output (with support counts as tuples) ----
with open(OUT_TXT, "w") as f:
    for support, pattern in patterns:
        # Final check to ensure no invalid patterns are written
        valid_pattern = all(is_valid_code(str(p)) for p in pattern)
        if valid_pattern and pattern:  # Only write valid, non-empty patterns
            f.write(f"({support}, {pattern}),\n")
 
print(f"Saved {len(patterns)} cleaned patterns to {OUT_TXT}")
 
# ---- Optional: Also create a Python list format ----
formatted_patterns = []
for support, pattern in patterns:
    valid_pattern = all(is_valid_code(str(p)) for p in pattern)
    if valid_pattern and pattern:
        formatted_patterns.append((support, pattern))
 
# Print first few patterns to verify format
print("\nFirst 10 cleaned patterns:")
for i, (support, pattern) in enumerate(formatted_patterns[:10]):
    print(f"({support}, {pattern}),")
 
print(f"\nTotal valid patterns found: {len(formatted_patterns)}")
 
# Show unique codes that were kept (for verification)
all_valid_codes = set()
for seq in sequences_cleaned:
    all_valid_codes.update(seq)
 
print(f"\nTotal unique valid codes: {len(all_valid_codes)}")
print("Sample valid codes:", sorted(list(all_valid_codes))[:20])