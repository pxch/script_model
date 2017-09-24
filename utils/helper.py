import os


def supress_fd(fd_number):
    assert fd_number in [1, 2]
    # open a null file descriptor
    null_fd = os.open(os.devnull, os.O_RDWR)
    # save the current stdout(1) / stderr(2) file descriptor
    save_fd = os.dup(fd_number)
    # put /dev/null fd on stdout(1) / stderr(2)
    os.dup2(null_fd, fd_number)

    return null_fd, save_fd


def restore_fd(fd_number, null_fd, save_fd):
    # restore stdout(1) / stderr(2) to the saved file descriptor
    os.dup2(save_fd, fd_number)
    # close the null file descriptor
    os.close(null_fd)
