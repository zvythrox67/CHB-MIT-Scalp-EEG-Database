import re
from pathlib import Path 

def parse_summary_file(summary_path):
    text = Path(summary_path).read_text(errors="ignore")
    blocks = re.split(r"(?=File Name:\s*chb)", text)
    
    seizures_by_file = {}
    
    for block in blocks:
        name_match = re.search(r"File Name:\s*(chb\S+\.edf)", block)
        if not name_match:
            continue
        fname = name_match.group(1)
        
        starts = [int(x) for x in re.findall(r"Seizure(?:\s*\d*)\s*Start Time:\s*(\d+)\s*seconds", block)]
        ends = [int(x) for x in re.findall(r"Seizure(?:\s*\d*)\s*End Time:\s*(\d+)\s*seconds", block)]
        
        intervals = list(zip(starts, ends))
        seizures_by_file[fname] = intervals
        
    return seizures_by_file