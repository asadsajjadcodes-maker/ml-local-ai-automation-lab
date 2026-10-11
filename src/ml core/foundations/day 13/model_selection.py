#======================================================================
#       Day 13 : Model Selection
#======================================================================


# Example evaluation results for three models.
models = {
    "Model A": {
        "train_mse": 80,
        "validation_mse":90
    },
    "Model B": {
        "train_mse":20,
        "validation_mse": 35
    },
    "Model C": {
        "train_mse": 2,
        "validation_mse": 150
    }
}

# Display model results,
for name, results in models.items():
    print(name)
    print(f"Training_mse: {results["train_mse"]}")
    print(f"Validation_mse : {results["validation_mse"]}")
    print("================================================")



# Select the model with the lowest validation MSE.
best_model = min(
    models,
    # this lambda function returns the validation MSE for each model name,
    #  allowing min() to find the model with the lowest validation MSE
    key= lambda name: models[name]["validation_mse"] 
)

print(f"Best model: {best_model}")

print(f"Best validation MSE: {models[best_model]["validation_mse"]}")

