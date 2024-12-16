from Board import Board


def main():
    board = Board()
    print(board)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")
