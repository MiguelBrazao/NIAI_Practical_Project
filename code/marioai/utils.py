import numpy


__all__ = ['extractObservation', 'Observation']

class Observation(object):
    """
    Structured observation data from the environment.
    """
    def __init__(self, may_jump=False, on_ground=False, mario_pos=None, enemies=None, 
                 level_scene=None, status=0, distance=0.0, time_left=0, mario_mode=0, coins=0):
        self.may_jump = may_jump
        self.on_ground = on_ground
        self.mario_pos = mario_pos
        self.enemies = enemies if enemies is not None else []
        self.level_scene = level_scene
        self.status = status
        self.distance = distance
        self.time_left = time_left
        self.mario_mode = mario_mode
        self.coins = coins

    def __repr__(self):
        return (f"Observation(may_jump={self.may_jump}, on_ground={self.on_ground}, "
                f"mario_pos={self.mario_pos}, enemies_len={len(self.enemies) if self.enemies else 0}, "
                f"status={self.status}, distance={self.distance})")

powsof2 = (1, 2, 4, 8, 16, 32, 64, 128,
                         256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072)

def decode(estate):
    """
    decodes the encoded state estate, which is a string of 61 chars
    """
    dstate = numpy.empty(shape = (22, 22), dtype = numpy.int32)
    for i in range(22):
        for j in range(22):
            dstate[i, j] = 2
    row = 0
    col = 0
    totalBitsDecoded = 0
    reqSize = 31
    assert len(estate) == reqSize, "Error in data size given %d! Required: %d \n data: %s " % (len(estate), reqSize, estate)
    
    for i in range(len(estate)):
        cur_char = estate[i]
        
        for j in range(16):
            totalBitsDecoded += 1
            if (col > 21):
                row += 1
                col = 0
            if ((int(powsof2[j]) & int(ord(cur_char))) != 0):
                dstate[row, col] = 1
            else:
                dstate[row, col] = 0
            col += 1
            if (totalBitsDecoded == 484):
                break
    return dstate

def extractObservation(data):
    """
    Parses the array of strings and returns an Observation namedtuple.
    """
    if isinstance(data, bytes):
        data = data.decode()

    # Handle fast TCP 'E' message (Merged observation)
    if data[0] == 'E':
        mayMarioJump = (data[1] == '1')
        isMarioOnGround = (data[2] == '1')
        levelScene = decode(data[3:34])
        # data[34:] checks sum but we can skip strict check for speed or implement if needed
        return Observation(may_jump=mayMarioJump, on_ground=isMarioOnGround, level_scene=levelScene)

    parts = data.split(' ')
    if parts[0] == 'FIT':
        status = int(parts[1])
        distance = float(parts[2])
        timeLeft = int(parts[3])
        marioMode = int(parts[4])
        coins = int(parts[5])
        # specific handling for fitness packet
        return Observation(status=status, distance=distance, time_left=timeLeft, mario_mode=marioMode, coins=coins)

    elif parts[0] == 'O':
        mayMarioJump = (parts[1] == 'true')
        isMarioOnGround = (parts[2] == 'true')
        
        # Parse Level Scene (22x22 flattened)
        # However, 'O' message format in original MarioAI is usually:
        # [O, mayJump, onGround, ...grid..., marioX, marioY, ...enemies...]
        # The original code looped 484 times.
        
        levelScene = numpy.empty(shape=(22, 22), dtype=numpy.int32)
        k = 0
        current_idx = 3
        for i in range(22):
            for j in range(22):
                levelScene[i, j] = int(parts[current_idx])
                current_idx += 1
        
        # Skip next ? (was k+=3 in original, but let's see)
        # Original: k initialized to 0. 22*22 = 484 iterations.
        # data[k+3] used. 
        # After loop, k = 484.
        # k += 3 -> k = 487.
        # marioFloats at data[487], data[488].
        
        # In my index variable `current_idx`:
        # Starts at 3. Ends at 3 + 484 = 487.
        # So next item is indeed at 487?
        # Original code did `k+=3` after loop.
        # Why? maybe separating chars or something?
        # Let's assume standard format: spaces between everything.
        # If the loop consumed 484 tokens, we are at index 3+484 = 487.
        # Original code: `k` was counting items read *from the grid*.
        # `data[k+3]`. Last read was k=483 -> `data[486]`.
        # Next index to read would be 487.
        # But original code said `k += 3`. So k becomes 487.
        # Access `data[487+3]` ?? No. `float(data[k])` where k is now 487?
        # Wait, original:
        # k=0. loop ... k=484.
        # k+=3 -> k=487.
        # floats = data[k], data[k+1] -> data[487], data[488].
        # So yes, we skip 3 items?
        # Actually, let's look at `extractObservation` original logic carefully.
        # `k` was the loop counter for grid cells.
        # `levelScene` filled from `data[k+3]`.
        # After loop, k=484.
        # `k += 3`. k=487.
        # `marioFloats` from `data[487]`.
        # This implies parts[487] is mario X.
        # It implies parts[3] through parts[486] are the grid.
        # This matches 484 items.
        
        # But what about the `k+=3` skip?
        # Maybe the grid is followed by something else before Mario pos?
        # Or maybe the data has some garbage?
        # Let's perform `current_idx = 487` directly to be safe?
        # No, `data` in original was split by ' '.
        # So indices matter.
        # Let's assume `current_idx` is now pointing to the item AFTER the grid.
        # Which is 487.
        # So if we mirror `k+=3`, we are skipping... wait.
        # Original: `k` is just an offset relative to something? No, `data[k+3]`.
        # If `k` goes from 0 to 483.
        # `data[3]` ... `data[486]`.
        # Then `k=487` refers to `data[487]`.
        # So using `data[k]` requires `k` to supply the full index? No, `data` is the array.
        # In the grid loop it was `data[k+3]`.
        # In the float section it is `data[k]`.
        # So `k` effectively changes semantic from "offset" to "absolute index".
        # If k became 487, it effectively "caught up" to being an index?
        # No, `k` was 484. `k+=3` -> 487.
        # If `data[k+3]` scans `data[3..486]`.
        # Then we want `data[487]`.
        # So `k` (which is 487) is used as index `data[487]`.
        # This is correct.
        
        marioFloats = (float(parts[487]), float(parts[488]))
        
        # k += 2 -> 489
        current_idx = 489
        enemiesFloats = []
        while current_idx < len(parts):
             # basic protection against empty strings at the end
            if parts[current_idx]:
                enemiesFloats.append(float(parts[current_idx]))
            current_idx += 1

        enemiesFloats = [ (enemiesFloats[i], enemiesFloats[i+1], enemiesFloats[i+2]) for i in range(0, len(enemiesFloats), 3) ]
        
        return Observation(may_jump=mayMarioJump, 
                           on_ground=isMarioOnGround, 
                           mario_pos=marioFloats, 
                           enemies=enemiesFloats, 
                           level_scene=levelScene)

    else:
        # Fallback or error
        # print("Unknown data format:", data)
        return Observation()

