state_history = [[1,2],[2,3],[3,4],[4,5],[5,6],[6,7],[7,8],[8,9]]

new_state = [1,2]


if new_state in state_history:
    print("True")
    while new_state in state_history:
        print("Invalid move")
        a,b = tuple(int(x.strip()) for x in input("\nInput your move: ").split(' '))

        new_state = [a,b]

state_history.append(new_state)
print(state_history)