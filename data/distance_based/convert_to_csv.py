import xml.etree.ElementTree as ET
import csv

def get_features_in_config(row):
    config = row.find('data[@column="Configuration"]').text.strip().split(',')
    if '' in config:
        config.remove('')
    return config
    
def convert_and_save(casestudy, targets):
    tree = ET.parse(f'{casestudy}/measurements.xml')
    root = tree.getroot()

    # Extract features
    features = set()
    for row in root.findall('row'):
        features.update(get_features_in_config(row))
    features = sorted(features)
    headers = features + targets

    # Write to CSV
    with open(f'{casestudy}/measurements.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=';')
        writer.writerow(headers)
        
        for row in root.findall('row'):
            row_data = [1 if feature in get_features_in_config(row) else 0 for feature in features]
            for target in targets:
                value = row.find(f'data[@column="{target}"]').text.strip()
                row_data.append(value)
            writer.writerow(row_data)

if __name__ == "__main__":
    casestudies = {
        '7z': ['Performance', 'Size'], 
        'BerkeleyDBC': ['Performance'], 
        'Dune': ['Performance'], 
        'Hipacc': ['Performance'], 
        'JavaGC': ['GC Time'], 
        'LLVM': ['Performance'], # There is a column MainMemory, but I don't know what it is 
        'lrzip': ['Performance'], 
        'Polly': ['Performance', 'ElapsedTime'], 
        'x264': ['Performance']
    }
    for casestudy, targets in casestudies.items():
        convert_and_save(casestudy, targets)