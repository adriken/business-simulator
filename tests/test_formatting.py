from simulator.formatting import input_label, money


def test_money():
    assert money(1_234_567) == "1.23M"
    assert money(45_600) == "46k"
    assert money(-2_500) == "-2k"
    assert money(12) == "12"


def test_input_label():
    assert input_label("streams.rooms.demand") == "rooms: demand"
    assert input_label("costs.fixed.staff") == "staff (fixed cost)"
    assert input_label("costs.per_unit.feed") == "feed (per-unit cost)"
    assert input_label("capacity.room_nights") == "capacity: room nights"
    assert input_label("streams.rooms.price:multiply[0]") == "rooms: price (plan change)"
