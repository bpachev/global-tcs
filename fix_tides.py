import sys
import glob

def set_tides(fort15, tides):
    with open(fort15) as fp:
        lines = fp.readlines()

    for i in range(len(lines)-1):
        l = lines[i].strip().split()[0]
        if l in tides:
            factor, deg = tides[l]
            next_line = lines[i+1]
            parts = next_line.split("!")
            data = parts[0].split()
            if len(data) < 3:
                continue
            data[-2], data[-1] = factor, deg
            stripped = parts[0].rstrip()
            trailing_whitespace = parts[0][len(stripped):]
            parts[0] = " ".join(data) + trailing_whitespace

            lines[i+1] = "!".join(parts)

    with open(fort15, "w") as fp:
        fp.write("".join(lines))

SUPPORTED_TIDES = {"Q1", "O1", "P1", "K1", "N2", "M2", "S2", "K2"}

def get_tides(fort15):
  with open(fort15) as fp:
    lines = fp.readlines()

  res = {}
  for i in range(len(lines)-1):
    l = lines[i].strip()
    if l in SUPPORTED_TIDES:
      if l in res: continue
      data = lines[i+1].split("!")[0].strip().split()
      res[l] = data[-2:]
  return res


if __name__ == "__main__":
  dirname = sys.argv[1]
  tides = get_tides(dirname+"/fort.15")
  for subfname in glob.glob(dirname+"/PE*/fort.15"):
    set_tides(subfname, tides)


