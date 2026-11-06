#==========================================================================================
#        Day 12 : Overfitting vs Underfitting
#==========================================================================================

def diagnose_model(train_mse, test_mse):
    print(f"Training MSE: {train_mse}")
    print(f"Testing MSE: {test_mse}")


    if train_mse > 10 and test_mse > 10 :
        print("Diagnosis: Possible Underfitting.")

    elif test_mse > train_mse * 3:
        print("Diagnosis: Possible Overfitting.")

    else:
        print("Diagnosis: Reasonably balanced.")

        print("=" * 10)


# Model A : Performs badly on both datasets 
print("Model A")
diagnose_model(train_mse=100, test_mse=120)

# Model B : Memorizes training data, struggles on test data 
print("Model B")
diagnose_model(train_mse=1, test_mse=100)

# Model C : Performs well on both datasets 
print("Model C")
diagnose_model(train_mse=5, test_mse=7)




