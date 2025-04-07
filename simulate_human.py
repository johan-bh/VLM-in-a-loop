import xml.etree.ElementTree as ET

# Path to your XML file (adjusted for Windows)
xml_file_path = r"C:\Users\jbhan\Desktop\VLM-in-a-loop\data\ecgen-radiology\1.xml"

def extract_binary_label(xml_path):
    """
    Extract a binary label from the IMPRESSION field.
    Returns 0 if normal, 1 if abnormal.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Find the IMPRESSION field
    impression = root.find('.//AbstractText[@Label="IMPRESSION"]')
    if impression is not None and impression.text:
        impression_text = impression.text.lower()
        if "normal" in impression_text:
            return 0  # Normal
        else:
            return 1  # Abnormal
    else:
        raise ValueError("No valid IMPRESSION field found.")

def simulate_human_response(predicted_label):
    """
    Simulate a human annotator who verifies the predicted label.
    """
    human_input = input(f"Predicted label is {'Normal' if predicted_label == 0 else 'Abnormal'}. Do you agree? (y/n): ")
    if human_input.lower() == 'y':
        return predicted_label
    else:
        return 1 - predicted_label  # flip the label if human disagrees

def main():
    # Extract binary label from XML
    label = extract_binary_label(xml_file_path)
    print(f"Extracted Label: {'Normal' if label == 0 else 'Abnormal'}")

    # Simulate human verification
    final_label = simulate_human_response(label)
    print(f"Final Label after Human Verification: {'Normal' if final_label == 0 else 'Abnormal'}")

if __name__ == "__main__":
    main()
