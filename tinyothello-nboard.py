from tinyothello import Position, start_pos, Searcher, Move, eval_all_moves

logf = open('nboardlog.txt', 'w')

def parse_ggf(str):
    #First setup the initial position
    i = str.find("BO[8 ")
    i += 5
    board = "          "
    for r in range(8):
        board += " "
        for c in range(8):
            board += {'*':'X', 'O':'O', '-':'.'}[str[i]]
            i += 1
        board += " "
    board += "          "
    i+=1
    side = 'X' if str[i]=='*' else 'O'
    i+=1
    position = Position(board, side)

    #Then play the moves
    split = str[i:].split(']')
    print(split)
    for substr in split:
        if substr == '': continue
        elif substr[0] == 'B':
            position.side = 'X'
        elif substr[0] == 'W':
            position.side = 'O'
        else: continue

        col = ord(substr[2]) - ord('A') + 1
        row = ord(substr[3]) - ord('1') + 1
        legal, flip = position.is_legal_move(row, col)
        if legal:
            position = position.make_move(Move(row, col, flip))

    return position

depth = -1
position = Position(start_pos, 'X')
searcher = Searcher()
while True:
    cmd_str = input()
    logf.write('>' + cmd_str + '\n')
    cmd = cmd_str.split()
    if cmd[0] == 'nboard':
        pass 
    elif cmd[0] == 'exit':
        break
    elif cmd[0] == 'set' and cmd[1] == 'depth':
        depth = int(cmd[2])
        print(f"set myname TinyOthello{depth}", flush=True)
    elif cmd[0] == 'set' and cmd[1] == 'game':
        position = parse_ggf(cmd_str)
    elif cmd[0] == 'ping':
        print('pong', cmd[1], flush=True)
    elif cmd[0] == 'go':
        print("status thinking", flush=True)
        print("nodestats 35805 0.00", flush=True)
        print("=== f5 -1.00 0.0", flush=True)
        print("status waiting", flush=True)
    elif cmd[0] == 'hint':
        print("status thinking", flush=True)
        result, nodes = eval_all_moves(position, depth, searcher)
        print(f"nodestats {nodes} 0.00", flush=True)
        for move, eval in result:
            print(f"search {'.ABCDEFGH'[move.col]}{move.row} {eval} 0 {depth} ff", flush=True)
        print("status waiting", flush=True)
    elif cmd[0] == 'move':
        col = ord(cmd[1][0]) - ord('A') + 1
        row = int(cmd[1][1])
        legal, flip = position.is_legal_move(row, col)
        if legal:
            position = position.make_move(Move(row, col, flip))
 
    
logf.close()