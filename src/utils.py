def list_pop_swap(list_, i):
    """Удаление элемента из списка без сохранения порядка."""
    assert i >= 0
    assert i < len(list_)

    list_[i] = list_[-1]
    list_.pop()
