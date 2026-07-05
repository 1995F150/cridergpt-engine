# Calculator tool for CriderGPT Engine
def calculate(expression: str):
    """Evaluates mathematical expressions."""
    try:
        return eval(expression)
    except Exception as e:
        return str(e)
