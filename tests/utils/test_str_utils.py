import cell_type_constellations.utils.str_utils as str_utils

def test_sanitization():
     assert str_utils.sanitize_taxon_name("a/b/c/d") == (
         "a$SLASH$b$SLASH$c$SLASH$d"
     )
     assert str_utils.sanitize_taxon_name("abcde") == "abcde"
