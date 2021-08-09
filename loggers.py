import time


class OutputLogger():
    def __init__(self, stream, log_file):
        self.stream = stream
        self.log = open(log_file, 'a+')

    def write(self, data):
        self.stream.write(data)
        self.log.write(data)

    def flush(self):
        self.stream.flush()


class TimeLogger():
    def __init__(self):
        self.start_time = time.time()

    def __call__(self):
        t = time.time() - self.start_time
        return round(t / 60, 2)

    def print(self):
        t = time.time() - self.start_time
        print('Time in seconds: {:.2f}'.format(t))
        print('Time in minutes: {:.2f}'.format(t / 60))


