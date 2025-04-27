import pandas as pd
from mlxtend.frequent_patterns import fpgrowth
from itertools import chain, combinations
from collections import Counter
import csv

data = pd.read_csv('./dataset/fine_food_combinations_01.csv')

data['items'] = data['items'].apply(lambda x: x.split(', '))
transactions = data['items'].tolist()

# calculate weighted support
def calculate_weighted_support(transactions, itemset):
    total_count = 0
    for t in transactions:
        item_counts = Counter(t)  # Count occurrences of each item in the transaction
        # Calculate the minimum occurrence count in this transaction
        min_count = min(item_counts.get(item, 0) for item in itemset)
        total_count += min_count
    # Calculate weighted support
    weighted_support = total_count / len(transactions)
    return weighted_support

# one-hot encoded DataFrame
oht = data['items'].str.join('|').str.get_dummies()

# Apply the FP-Growth algorithm
frequent_itemsets = fpgrowth(oht, min_support=0.001, use_colnames=True)

# Convert itemsets to frozenset
frequent_itemsets['itemsets'] = frequent_itemsets['itemsets'].apply(lambda x: frozenset(x))

# Calculate weighted support for frequent itemsets
frequent_itemsets['weighted_support'] = frequent_itemsets['itemsets'].apply(lambda x: calculate_weighted_support(transactions, x))

# calculate confidence
def calculate_confidence(antecedent, consequent, frequent_itemsets):
    # Ensure antecedents and consequents are frozensets
    if not isinstance(antecedent, frozenset):
        antecedent = frozenset(antecedent)
    if not isinstance(consequent, frozenset):
        consequent = frozenset(consequent)

    # Calculate support for antecedent and consequent
    support_antecedent = frequent_itemsets[frequent_itemsets['itemsets'] == antecedent]['weighted_support'].values
    support_consequent = frequent_itemsets[frequent_itemsets['itemsets'] == consequent]['weighted_support'].values

    # Calculate support for the combination of antecedent and consequent
    support_union = frequent_itemsets[frequent_itemsets['itemsets'] == antecedent.union(consequent)]['weighted_support'].values

    # Check if the arrays are not empty before accessing its values
    if support_antecedent.size > 0 and support_union.size > 0:
        confidence = support_union[0] / support_antecedent[0]
    else:
        confidence = 0

    return confidence

# generate all non-empty proper subsets
def all_nonempty_proper_subsets(itemset):
    s = list(itemset)
    return chain.from_iterable(combinations(s, r) for r in range(1, len(s)))

# Generate association rules
rules_list = []
for i in range(len(frequent_itemsets)):
    itemset = frequent_itemsets['itemsets'][i]
    if len(itemset) > 1:
        for antecedent in map(frozenset, all_nonempty_proper_subsets(itemset)):
            consequent = itemset.difference(antecedent)
            confidence = calculate_confidence(antecedent, consequent, frequent_itemsets)
            rules_list.append({
                'antecedents': antecedent,
                'consequents': consequent,
                'Coarse_Class_Confidence': confidence
            })

# Convert to output DataFrame
output = pd.DataFrame(rules_list)

# Filter rules by minimum confidence threshold
min_confidence = 0.001
output = output[output['Coarse_Class_Confidence'] >= min_confidence]

output['antecedents'] = output['antecedents'].apply(lambda x: ', '.join(list(x)))
output['consequents'] = output['consequents'].apply(lambda x: ', '.join(list(x)))

output.columns = ['Coarse_Class_Antecedent', 'Coarse_Class_Consequent', 'Coarse_Class_Confidence']

# Save to CSV without weighted_support column
output.to_csv('./process/Coarse_association_rules_output_01.csv', index=False, header=True, quoting=csv.QUOTE_NONNUMERIC)

print("Association rules have been saved to 'Coarse_association_rules_output_01.csv' with headers.")
