import logging

INT16_MAX = int("7FFF", 16)
INT16_MIN = int("-8000", 16)


log = logging.getLogger()
logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)


def list_pop_swap(list_, i):
    """Удаление элемента из списка без сохранения порядка."""
    assert i >= 0
    assert i < len(list_)

    list_[i] = list_[-1]
    list_.pop()
