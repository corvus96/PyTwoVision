from recognition import Recognizer
from recognition import NeuralNetwork

class SimultaneousNetwork(NeuralNetwork):
    def train_net(self):
        return "ConcreteImplementationA: Here's the result on the platform A."


def client_code(recognizer: Recognizer):
    """
    Except for the initialization phase, where an Abstraction object gets linked
    with a specific Implementation object, the client code should only depend on
    the Abstraction class. This way the client code can support any abstraction-
    implementation combination.
    """

    # ...

    print(recognizer.train(), end="")

    # ...


if __name__ == "__main__":
    """
    The client code should be able to work with any pre-configured abstraction-
    implementation combination.
    """

    implementation = ConcreteImplementationA()
    abstraction = Recognizer(implementation)
    client_code(abstraction)

    print("\n")
