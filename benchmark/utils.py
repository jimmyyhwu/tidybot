def print_colored(text, color='red', end='\n'):
    color_code = {
        'red': '\033[91m',
        'green': '\033[92m',
        'blue': '\033[34m',
    }[color]
    print(f'{color_code}{text}\033[00m', end=end)
