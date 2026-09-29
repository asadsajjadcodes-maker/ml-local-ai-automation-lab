hours = 6 
actual_score = 70

weight = 1
learning_rate = 0.01



for i in range(20):
    prediction = hours * weight

    error = actual_score - prediction

    loss = error ** 2

    gradient = -2 * hours * error

    weight = weight - learning_rate * gradient


    print(f"Step {i + 1}")
    print(f"Prediction : {prediction:.2f}")
    print(f"Error: {error:.2f}")
    print(f"Loss : {loss:.2f}")
    print(f"Weight: {weight:.2f}")
    print("=============================")