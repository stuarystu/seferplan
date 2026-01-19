import csv

NORMALCI_MAX_GAP = 90
TEKCI_MAX_GAP = 15
NORMALCI_MIN_SERVIS = 4

def to_min(t):
    h, m = t.split(":")
    return int(h) * 60 + int(m)

def max_gap(services):
    services = sorted(services, key=lambda x: to_min(x["GIDIS"]))
    mg = 0
    for i in range(len(services)-1):
        gap = to_min(services[i+1]["GIDIS"]) - to_min(services[i]["DONUS"])
        mg = max(mg, gap)
    return mg

def planla(rows):
    tekci = []
    normalci = []

    rows = sorted(rows, key=lambda x: to_min(x["GIDIS"]))

    for r in rows:
        placed = False

        for c in tekci:
            if max_gap(c + [r]) <= TEKCI_MAX_GAP:
                c.append(r)
                placed = True
                break
        if placed:
            continue

        for c in normalci:
            if len(c) >= NORMALCI_MIN_SERVIS and max_gap(c + [r]) <= NORMALCI_MAX_GAP:
                c.append(r)
                placed = True
                break
        if placed:
            continue

        tekci.append([r])

    normalci = [c for c in normalci if len(c) >= NORMALCI_MIN_SERVIS]
    return tekci, normalci
