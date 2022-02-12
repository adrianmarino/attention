from error.common_exception import CommonException


class Expression:
    def __init__(self, expression, message):
        self.__expression = expression
        self.message = message

    def exec(self, value):
        return self.__expression(value)


class Checker:
    def __init__(self, error_code, value, name='Value'):
        self.__expressions = []
        self.__error_code = error_code
        self.__value = value
        self.__name = name

    @staticmethod
    def positive_int(error_code, value, name):
        Checker(error_code, value, name).is_not_none().is_int().is_positive().check()

    @staticmethod
    def positive_float(error_code, value, name):
        Checker(error_code, value, name).is_not_none().is_float().is_positive().check()

    def is_not_none(self):
        self.__expressions.append(Expression(lambda it: it is not None, f'{self.__name} is None'))
        return self

    def is_int(self):
        self.__expressions.append(Expression(lambda it: isinstance(it, int), f'{self.__name} is not int'))
        return self

    def is_float(self):
        self.__expressions.append(Expression(lambda it: isinstance(it, float), f'{self.__name} is not float'))
        return self

    def is_positive(self):
        self.__expressions.append(Expression(lambda it: it > 0, f'{self.__name} is not a positive number'))
        return self

    def check(self):
        errors = self.__errors()
        if errors:
            raise CommonException(self.__error_code, errors)

    def __errors(self):
        return [exp.message for exp in self.__expressions if not exp.exec(self.__value)]
