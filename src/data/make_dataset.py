from subprocess import call


def main():
    cmd = 'kaggle competitions download -c home-credit-default-risk --path data/raw'
    call(cmd, shell=True)
    
if __name__ == '__main__':
    main()
