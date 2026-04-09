from dt4acc_lib.bl.yellow_pages import YellowPages


def test_yellow_pages():

    quad_names = ["schott", "newport"]
    sext_names = ["zeiss", "leica"]
    yp = YellowPages(
        dict(quadrupoles=quad_names,
             sextupoles=sext_names
        )
    )

    names = yp.get("quadrupoles")
    assert len(names) == len(quad_names)
    for name in names:
        assert name in quad_names

    names = yp.get("sextupoles")
    assert len(names) == len(sext_names)
    for name in names:
        assert name in sext_names
