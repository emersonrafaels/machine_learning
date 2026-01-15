"""

CASE: WHETHER OR NOT OUR HOUSE WILL REQUIRE A NEW ROOF IN THE NEXT 10 YEARS.

TARGET:
    1 - ROOF-REPLACEMENT
    0 - NO ROOF-REPLACEMENT

"""

# IMPORT LIB
import vowpalwabbit


def load_init_data():

    # CREATING EXAMPLES - TRAIN
    train_examples = [
        "0 | price:.23 sqft:.25 age:.05 2006",
        "1 | price:.18 sqft:.15 age:.35 1976",
        "0 | price:.53 sqft:.32 age:.87 1924",
    ]

    return train_examples


def load_test_data():

    # CREATING EXAMPLES - TEST
    test_examples = ["| price:.46 sqft:.4 age:.10 1924"]

    return test_examples


# CREATING AN INSTANCE OF VOWPAL WABBIT
# quiet=True, avoid diagnostic informations in console
model = vowpalwabbit.Workspace(quiet=True)

# GET INIT DATA - TRAIN
train_init_examples = load_init_data()

# TRAINING WITH INIT DATA
[model.learn(example) for example in train_init_examples]

# GET - TEST
test_examples = load_test_data()

# DOING PREDICTIONS
predictions = [model.predict(example) for example in test_examples]
print(predictions)
