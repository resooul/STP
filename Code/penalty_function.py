
def penalty_function(target, prediction):
    if (target != prediction):
        return 10000000
    else:
        return 0