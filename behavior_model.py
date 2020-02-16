MODEL_TEMPLATE = '''
class MyBehaviorModel:
    relevant_inputs = [
        'My Input 1',
        'My Input 2'
    ]

    relevant_outputs = [
        'My Output 1',
        'My Output 2',
    ]

    def validate(self, inputs, outputs):
        """Returns whether the signals represent correct operation of the device
        """
        input1 = self.relevant_input_values[0]
        input2 = self.relevant_input_values[1]
        output1 = self.relevant_output_values[0]
        output2 = self.relevant_output_values[1]

        raise NotImplementedError

ModelClass = MyBehaviorModel
'''

class BehaviorModel:
    relevant_inputs = [
        'Input 1',
        'Input 2'
    ]

    relevant_outputs = [
        'Output 1',
        'Output 2',
    ]

    def validate(self, inputs, outputs):
        """Returns whether the signals represent correct operation of the device
        """
        return True
