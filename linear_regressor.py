from tqdm import tqdm

class LinearRegressor:
    m = [float] # weights
    
    def __init__(self, m=[float]):
        '''
        initialize a liniear regression function
        y = m0 + m1x1 + m2x2 + ... + mnxn
        x: [1, x1, x2, ..., xn]
        m: [m0, m1, m2, ..., mn]
        '''
        self.m = m
        
    def predict(self, x):
        '''
        predict y value from x
        y = m0 + m1x1 + m2x2 + ... + mnxn
        '''
        y = 0
        n = len(self.m)
        for i in range(n):
            y += self.m[i] * x[i]
        return y
    
    def fit(self, X: list[list[float]], y: list, lr=0.01, epochs=10, epsilon=0.1, m:[float]=None):
        '''
        fit the model with X and y using gradient descent
        X: [[1, x1, x2, ..., xn], ...]
        y: [y1, y2, ..., yn]
        lr: learning rate -> float
        epsilon: error threshold -> float
        epochs: number of epochs -> int
        ---
        updates self.m
        '''
        if len(X) != len(y):
            raise Exception('X and y must have same length')
        
        n = len(X)
        # Create X matrix with 1s as first column
        X = [[1] + X[i] for i in range(n)]
        
        # Create initial m weights
        if (m != None):
            self.m = m
        else:
            self.m = [1 for i in range(len(X[0]))]
        # Gradient Descent
        progress_bar = tqdm(total=epochs, desc="Training Progress")
        error_history = []
        for epoch in range(epochs):
            # Calculate error
            y_pred = [self.predict(X[i]) for i in range(n)]
            e = LinearRegressor.loss(y_pred, y)
            
            error_history.append(e)
            # Calculate gradient
            grad = [0 for i in range(len(X[0]))]
            for i in range(n):
                for j in range(len(X[0])):
                    grad[j] += (y_pred[i] - y[i]) * X[i][j]
            for j in range(len(X[0])):
                grad[j] = grad[j] / n
            
            # Update m
            for j in range(len(X[0])):
                self.m[j] = self.m[j] - lr * grad[j]
            
            
            # Update progress bar description
            progress_bar.set_description(f"Epoch: {epoch} | Error: {e:.4f}")
            progress_bar.update(1)
            
            # Check error threshold
            if e < epsilon:
                print(f'Error threshold reached at epoch {epoch}')
                break

        progress_bar.close()
        print(f'X: {X}\nm: {self.m}')
        
        return m, error_history
    
    def loss(y_pred, y_true):
        '''
        Mean Squared Error
        e = (1/n) * (sigma: i~n) (y_pred[i] - y_true[i]) ** 2
        '''
        if len(y_pred) != len(y_true):
            raise Exception('y_pred and y_true must have same length')
        
        e = 0
        n = len(y_pred)
        for i in range(n):
            e += (y_pred[i] - y_true[i]) ** 2
        return e / n