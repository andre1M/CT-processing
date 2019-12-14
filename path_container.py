import os


class PathContainer:
    def __init__(self, stack_dir, ref_stack_dir, resulting_stack_dir, output_dir):
        """
        Store all required paths
        """
        self.stack = os.path.dirname(os.path.abspath(__file__)) + '/' + stack_dir + '/'
        self.ref_stack = os.path.dirname(os.path.abspath(__file__)) + '/' + ref_stack_dir + '/'
        self.resulting_stack = os.path.dirname(os.path.abspath(__file__)) + '/' + resulting_stack_dir + '/'
        self.output = os.path.dirname(os.path.abspath(__file__)) + '/' + output_dir + '/'
