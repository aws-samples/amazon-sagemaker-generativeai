from typing import Dict, List, Optional
import json

# Simple knowledge base for demonstration
CONCEPT_KB = {
    "elephant": {
        "properties": ["large", "heavy", "loud", "grey", "strong"],
        "category": "animal",
        "size": 5,  # Scale 1-5
        "volume": 5,  # Scale 1-5
        "weight": 5,  # Scale 1-5
    },
    "mouse": {
        "properties": ["small", "light", "quiet", "grey", "quick"],
        "category": "animal",
        "size": 1,
        "volume": 1,
        "weight": 1,
    },
    "car": {
        "properties": ["metallic", "fast", "loud", "heavy"],
        "category": "vehicle",
        "size": 4,
        "volume": 4,
        "weight": 4,
    }
}

def get_related(concept: str) -> str:
    """Get related properties and information about a concept."""
    concept = concept.lower().strip()
    if concept not in CONCEPT_KB:
        return f"No information available about {concept}"
    
    info = CONCEPT_KB[concept]
    return json.dumps({
        "properties": info["properties"],
        "category": info["category"]
    }, indent=2)

def compare(concept1: str, concept2: str, attribute: Optional[str] = None) -> str:
    """Compare two concepts, optionally on a specific attribute."""
    concept1, concept2 = concept1.lower().strip(), concept2.lower().strip()
    
    if concept1 not in CONCEPT_KB:
        return f"No information available about {concept1}"
    if concept2 not in CONCEPT_KB:
        return f"No information available about {concept2}"
    
    info1, info2 = CONCEPT_KB[concept1], CONCEPT_KB[concept2]
    
    if attribute:
        if attribute not in info1 or attribute not in info2:
            return f"Cannot compare {attribute} for these concepts"
        diff = info1[attribute] - info2[attribute]
        return json.dumps({
            "difference": diff,
            f"{concept1}_{attribute}": info1[attribute],
            f"{concept2}_{attribute}": info2[attribute]
        }, indent=2)
    
    # Compare all common numeric attributes
    common_attrs = set(info1.keys()) & set(info2.keys())
    numeric_comparisons = {}
    for attr in common_attrs:
        if isinstance(info1[attr], (int, float)) and isinstance(info2[attr], (int, float)):
            numeric_comparisons[attr] = {
                concept1: info1[attr],
                concept2: info2[attr],
                "difference": info1[attr] - info2[attr]
            }
    
    return json.dumps(numeric_comparisons, indent=2) 