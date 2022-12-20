class History:
    """
    Store data in a history with a maximum size after which the oldest entry is
    discarded
    """
    def __init__(self, size=1):
        """
        :param size: How many entries should be kept in this history
        """
        self.history_size = size
        self.entries = []

    def reset(self):
        """
        Deletes all saved entries in the history
        :return:
        """
        self.entries = []

    def add(self, new_entry):
        """
        Adds an entry to the history
        :param new_entry: Entry that should be saved in the history
        :return:
        """
        self.entries.append(new_entry)
        # don't exceed the maximum history size
        if len(self.entries) > self.history_size:
            self.entries.pop(0)

    def most_recent(self):
        """
        Returns the most recently added entry from the history
        :return: Most recent observation
        """
        return self.entries[-1]

    def is_empty(self):
        """
        Returns true, if the history is empty
        :return: True if history empty, else false
        """
        return True if len(self.entries) == 0 else False