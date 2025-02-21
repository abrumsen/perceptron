from random import gauss
import pandas as pd

LEARNING_DATASET = "./datasets/and_gate.csv"
MU = 0
SIGMA = 1.2
ETA = 1
MAX_TRAINING = 5000


class Perceptron:
    """
    A simple perceptron model for binary classification.
    This class implements training, prediction, and evaluation functionality.

    Attributes:
        synaptic_weights (list[float]): The weights used for making predictions,
            which get updated during training.
        l_inputs (list[list[int]]): The processed training inputs with a bias term
            added as the first element.
        labels (list[int]): The expected output values (ground truth) for the training inputs.
        learning_step (float): The learning rate controlling the magnitude of weight updates.
        max_learning (int): The maximum number of training iterations.
    """

    def __init__(self, synaptic_weights, examples, learning_step, max_learning) -> None:
        self.synaptic_weights = synaptic_weights
        self.l_inputs = [[1] + inputs[1:-1] for inputs in examples]
        self.labels = [label[-1] for label in examples]
        self.learning_step = learning_step
        self.max_learning = max_learning

    def guess(self, inputs: list[int]) -> int:
        """
        Makes a prediction based on the given inputs and the current synaptic weights.

        This method calculates the weighted sum of the inputs using the synaptic weights,
        and returns 1 if the sum is greater than or equal to 0, indicating activation,
        or 0 if the sum is less than 0, indicating lack of activation.

        Args:
            inputs (list[int]): A list of input values to be processed by the model.

        Returns:
            int: Activation (1 or 0) based on the weighted sum of the inputs.
        """
        potential = 0
        for i, input in enumerate(inputs):
            potential += self.synaptic_weights[i] * input
        if potential >= 0:
            return 1
        return 0

    def train(self) -> None:
        """
        Trains the model using a simple perceptron learning algorithm.

        This method iterates through the training dataset, making predictions using the
        `guess` method, calculating errors, and updating synaptic weights using the
        perceptron learning rule. Training stops early if no errors occur in an iteration.
        """
        print(
            f"\nStarting training with the following synaptic weights : {self.synaptic_weights}"
        )
        for iterations in range(self.max_learning):
            errors = 0
            for i, inputs in enumerate(self.l_inputs):
                output = self.guess(inputs)
                error = self.labels[i] - output
                if error:
                    errors += 1
                    for j in range(len(self.synaptic_weights)):
                        self.synaptic_weights[j] += (
                            self.learning_step * error * inputs[j]
                        )
            if errors == 0:
                print(
                    f"Finished training after {iterations} iterations with the following synaptic weights : {self.synaptic_weights}"
                )
                break

    def plot(self):
        # TODO
        pass

    def __call__(self) -> None:
        """
        Executes the training process and evaluates the model before and after training.

        This method is invoked when an instance of the class is called like a function.
        It prints the model’s predictions on the training dataset before and after
        training, showing how the synaptic weights adjust to improve accuracy.

        The process follows these steps:
        1. Print the model’s initial predictions for each input in `LEARNING_DATASET`.
        2. Train the model using the `train` method.
        3. Print the model’s predictions again after training to compare improvements.
        """
        print(f"\nBefore training on {LEARNING_DATASET}:")
        for i, inputs in enumerate(self.l_inputs):
            print(
                f"Example {i + 1}, Inputs: {inputs[1:]}, Outcome: {self.guess(inputs)}, Expected: {self.labels[i]}"
            )
        self.train()  # Insane training arc
        print(f"\nAfter training:")
        for i, inputs in enumerate(self.l_inputs):
            print(
                f"Example {i + 1}, Inputs: {inputs[1:]}, Outcome: {self.guess(inputs)}, Expected: {self.labels[i]}"
            )


def main() -> None:
    print(f"Reading the dataset located at {LEARNING_DATASET}")
    ld = pd.read_csv(LEARNING_DATASET)
    examples = ld.values.tolist()
    print(
        f"Initializing synaptic weights using the following distribution: gauss({MU},{SIGMA})"
    )
    synaptic_weights = [gauss(MU, SIGMA) for _ in range(len(examples[0]) - 1)]
    perceptron = Perceptron(synaptic_weights, examples, ETA, MAX_TRAINING)
    print(
        f"Created perceptron object with learning step of {ETA} and max training iterations of {MAX_TRAINING}"
    )
    perceptron()


if __name__ == "__main__":
    main()
