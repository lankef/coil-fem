def fetch_attr(func_name: str, module):
    if hasattr(module, func_name):
        attr = getattr(module, func_name)
    else:
        raise ValueError(f'\'{func_name}\' not found in {str(module)}')
    return attr

