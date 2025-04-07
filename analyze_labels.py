import xml.etree.ElementTree as ET
import os
import glob
from collections import Counter

# Path to XML files (adjusted for Windows)
xml_dir = r"C:\Users\jbhan\Desktop\VLM-in-a-loop\data\ecgen-radiology"

# Containers for different types of labels
abstract_labels = set()  # Store all unique Abstract section labels
mesh_major_labels = []   # Store all major MeSH terms
mesh_auto_labels = []    # Store all automatic MeSH terms
impression_labels = {"normal": 0, "abnormal": 0}  # Binary classification counts

# Process all XML files
xml_files = glob.glob(os.path.join(xml_dir, "*.xml"))
print(f"Found {len(xml_files)} XML files to process")

for i, xml_path in enumerate(xml_files):
    if i % 100 == 0:
        print(f"Processing file {i+1}/{len(xml_files)}")
        
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        # 1. Collect all Abstract section labels
        abstract_sections = root.findall('.//AbstractText')
        for section in abstract_sections:
            if 'Label' in section.attrib:
                abstract_labels.add(section.attrib['Label'])
        
        # 2. Extract major MeSH terms
        mesh_major = root.findall('.//MeSH/major')
        for term in mesh_major:
            if term.text:
                mesh_major_labels.append(term.text)
        
        # 3. Extract automatic MeSH terms
        mesh_auto = root.findall('.//MeSH/automatic')
        for term in mesh_auto:
            if term.text:
                mesh_auto_labels.append(term.text)
                
        # 4. Check if the impression is normal or abnormal
        impression = root.find('.//AbstractText[@Label="IMPRESSION"]')
        if impression is not None and impression.text:
            impression_text = impression.text.lower()
            if "normal" in impression_text:
                impression_labels["normal"] += 1
            else:
                impression_labels["abnormal"] += 1
                
    except Exception as e:
        print(f"Error processing {xml_path}: {e}")

# Print results
print("\n--- ABSTRACT SECTION LABELS ---")
print(f"Found {len(abstract_labels)} different abstract section labels:")
print(sorted(abstract_labels))

print("\n--- MAJOR MESH TERMS (TOP 30) ---")
major_counter = Counter(mesh_major_labels)
print(f"Found {len(major_counter)} unique major MeSH terms")
for term, count in major_counter.most_common(30):
    print(f"{term}: {count}")

print("\n--- AUTOMATIC MESH TERMS (TOP 30) ---")
auto_counter = Counter(mesh_auto_labels)
print(f"Found {len(auto_counter)} unique automatic MeSH terms")
for term, count in auto_counter.most_common(30):
    print(f"{term}: {count}")

print("\n--- BINARY CLASSIFICATION (NORMAL/ABNORMAL) ---")
print(f"Normal cases: {impression_labels['normal']}")
print(f"Abnormal cases: {impression_labels['abnormal']}")
print(f"Total cases with impression: {impression_labels['normal'] + impression_labels['abnormal']}")
print(f"Percentage normal: {impression_labels['normal'] / (impression_labels['normal'] + impression_labels['abnormal']) * 100:.2f}%") 