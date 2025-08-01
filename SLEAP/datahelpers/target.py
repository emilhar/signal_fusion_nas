class Target():
    def __init__(self, given_name, data_label):
        self.given_name = given_name
        self.data_label = data_label

    def __str__(self):
        return self.given_name
    