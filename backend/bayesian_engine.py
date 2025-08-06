from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

# ✅ Define a simple Bayesian Network structure
model = DiscreteBayesianNetwork([('BloomLevel', 'Mastery')])

# BloomLevel node: 0 = Low-level (Remember/Understand), 1 = High-level (Apply+)
cpd_bloom = TabularCPD(
    variable='BloomLevel', variable_card=2,
    values=[[0.6], [0.4]]
)

# Mastery node: 0 = Weak, 1 = Strong
cpd_mastery = TabularCPD(
    variable='Mastery', variable_card=2,
    values=[
        [0.8, 0.4],  # Weak
        [0.2, 0.6]   # Strong
    ],
    evidence=['BloomLevel'],
    evidence_card=[2]
)

# Add CPDs and validate model
model.add_cpds(cpd_bloom, cpd_mastery)
model.check_model()

# Create inference engine
inference = VariableElimination(model)

# ✅ Mastery Estimator Function
def estimate_mastery(bloom_level: str) -> float:
    """
    Estimate mastery score (0–1) based on Bloom level.
    """
    '''high_levels = ["apply", "analyze", "evaluate", "create"]
    level = 1 if bloom_level.lower() in high_levels else 0
    result = inference.query(variables=["Mastery"], evidence={"BloomLevel": level})
    return round(float(result.values[1]), 2)  # Probability of Strong Mastery'''
    high_levels = ["apply", "analyze", "evaluate", "create"]
    low_levels = ["remember", "understand"]
    level_lower = bloom_level.lower()
    
    if level_lower in high_levels:
        level = 1
    elif level_lower in low_levels:
        level = 0
    else:
        print(f"⚠️ Warning: Unknown Bloom level '{bloom_level}'. Defaulting to low-level (0).")
        level = 0
    
    result = inference.query(variables=["Mastery"], evidence={"BloomLevel": level})
    return round(float(result.values[1]), 2)  # P(Strong Mastery)

# 🧪 Optional test block
if __name__ == "__main__":
    test_levels = ["Remember", "Understand", "Apply", "Analyze", "Evaluate", "Create", "Unknown"]
    for level in test_levels:
        print(f"{level}: Mastery Score = {estimate_mastery(level)}")
