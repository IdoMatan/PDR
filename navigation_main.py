from navigation_utils import *


# -------------------- MAIN ------------------------------------------------------------------------

dir_name = 'walking/inhand-28-steps-Ido'


def main():
    dead_reckon(dir_name, remove_bias=False, title='Results plot with bias')
    dead_reckon(dir_name, remove_bias=True, title='Results plot without bias', sma=0)


if __name__ == "__main__":
    main()
