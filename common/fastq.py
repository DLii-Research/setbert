def read(buffer):
    lines = buffer.readlines()
    if len(lines) > 0 and type(lines[0]) is bytes:
        lines = [line.decode() for line in lines]
    return [FastqEntry.from_lines(lines[i:i+4]) for i in range(0, len(lines), 4)]


def parse_fastq_sequence_id(sequence_id):
    info = {}

    # Split up the sequence ID information
    left, right = sequence_id.strip()[1:].split(' ')
    left = left.split(':')
    right = right.split(':')

    info["instrument"] = left[0]
    info["run_number"] = int(left[1])
    info["flowcell_id"] = left[2]
    info["lane"] = int(left[3])
    info["tile"] = int(left[4])
    info["pos"] = tuple(map(int, left[5:]))

    info["read_type"] = int(right[0])
    info["is_filtered"] = right[1] == 'Y'
    info["control_number"] = int(right[2])
    info["sequence_index"] = right[3]

    return info


class FastqSequenceId:
    """
    A class representation of the sequence identifier of a FASTQ entry.
    """

    @classmethod
    def from_str(cls, sequence_id_str: str):
        info = parse_fastq_sequence_id(sequence_id_str)
        return cls(**info)

    def __init__(
        self,
        instrument,
        run_number,
        flowcell_id,
        lane,
        tile,
        pos,
        read_type,
        is_filtered,
        control_number,
        sequence_index,
    ):
        self.instrument = instrument
        self.run_number = run_number
        self.flowcell_id = flowcell_id
        self.lane = lane
        self.tile = tile
        self.pos = pos
        self.read_type = read_type
        self.is_filtered = is_filtered
        self.control_number = control_number
        self.sequence_index = sequence_index

    def encode(self):
        sequence_id = '@'
        sequence_id += ':'.join(map(str, [
            self.instrument,
            self.run_number,
            self.flowcell_id,
            self.lane,
            self.tile,
            *self.pos
        ]))
        sequence_id += ' '
        sequence_id += ':'.join(map(str, [
            self.read_type,
            'Y' if self.is_filtered else 'N',
            self.control_number,
            self.sequence_index
        ]))
        return sequence_id

    def __str__(self):
        return self.encode()

    def __repr__(self):
        return str(self)


class FastqEntry:
    """
    A class representation of a FASTQ entry containing the sequnce identifier, sequence, and quality
    scores.
    """

    @classmethod
    def from_lines(cls, lines):
        assert len(lines) == 4
        return cls(lines[1], lines[3], FastqSequenceId.from_str(lines[0]))

    @classmethod
    def from_buffer(cls, buffer):
        return cls.from_lines(buffer.readlines(4))

    def __init__(
        self,
        sequence: str,
        quality_scores: str,
        sequence_id = None
    ):
        self.sequence = sequence.strip()
        self.quality_scores = quality_scores.strip()
        self.sequence_id = sequence_id

    def __str__(self):
        return f"{self.sequence_id}\n{self.sequence}\n+\n{self.quality_scores}"

    def __repr__(self):
        return "FastqEntry:\n" + '\n'.join(f"  {s}" for s in str(self).split('\n'))
