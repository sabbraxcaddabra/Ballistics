class TooMuchPowderError(Exception):
    def __str__(self):
        return "Слишком много пороха"

class TooMuchTimeError(Exception):
    def __str__(self):
        return "Превышено максимальное время процесса"

class NoOneCombo(Exception):
    def __str__(self):
        return "С данной невязкой не найдено ни одной комбинации"