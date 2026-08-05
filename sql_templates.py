"""
Few-shot SQL templates untuk Qwen2.5-1.5B Text-to-SQL.

Format prompt yang digunakan:
- Tabel + kolom eksplisit (bukan schema abstrak)
- Contoh question→SQL per tabel
- Model tinggal meniru pola, tidak perlu "mengarang" table name

Schema: PostgreSQL, semua tabel ada di schema "global"
"""

# Keywords yang TIDAK ada di database dev_richz — jangan generate SQL untuk ini
# Data ini ada di database per-aplikasi lain (richzspot_dev, fjm_dev, dll)
UNSUPPORTED_KEYWORDS = [
    "best seller", "terlaris", "paling laku",
    "menu makanan", "menu minuman", "menu restoran", "daftar makanan",
    "stok", "stock", "harga produk",
    "invoice", "struk", "receipt",
    "penjualan", "sales",
    "pembelian", "purchase",
    "katalog", "catalogue",
    "logistik",
]

# Kata makanan/restoran — kalau ada di pertanyaan, query menu dianggap menu makanan (unsupported)
FOOD_CONTEXT_KEYWORDS = [
    "makanan", "minuman", "restoran", "warung", "cafe", "makan", "mie", "nasi",
    "ayam", "soto", "bakso", "pizza", "burger", "masakan", "hidangan",
]

# Mapping keyword → tabel yang relevan
TABLE_KEYWORDS = {
    "global_member": [
        "member", "pelanggan", "anggota", "customer", "poin", "reward"
    ],
    "global_auth_user": [
        "user", "pengguna", "pegawai", "karyawan", "login", "akun", "staff"
    ],
    "global_departemen": [
        "departemen", "department", "cabang", "dep", "divisi", "unit"
    ],
    "global_master_outlet": [
        "outlet", "toko", "gerai", "store", "lokasi"
    ],
    "global_shift": [
        "shift", "jadwal", "jam kerja"
    ],
    "global_menu": [
        "menu navigasi", "menu aplikasi", "fitur", "modul", "navigasi"
    ],
    "global_omset": [
        "omset", "pendapatan", "revenue", "penjualan", "sales"
    ],
    "global_system_activity_log": [
        "log", "aktivitas", "activity", "history", "riwayat", "audit"
    ],
}

# Few-shot examples per tabel: (question, sql)
FEW_SHOT_EXAMPLES = {
    "global_member": [
        ("Berapa jumlah member?",
         "SELECT COUNT(*) AS total_member FROM global.global_member WHERE deleted_at IS NULL"),
        ("Tampilkan 5 member terbaru",
         "SELECT member_nama, member_no_hp, member_email, member_poin FROM global.global_member WHERE deleted_at IS NULL ORDER BY member_create DESC LIMIT 5"),
        ("Siapa saja member aktif?",
         "SELECT member_nama, member_no_hp, member_poin FROM global.global_member WHERE member_aktif = '1' AND deleted_at IS NULL LIMIT 10"),
        ("Member dengan poin terbanyak",
         "SELECT member_nama, member_poin FROM global.global_member WHERE deleted_at IS NULL ORDER BY member_poin DESC LIMIT 10"),
        ("Cari member dengan nama Budi",
         "SELECT member_nama, member_no_hp, member_email FROM global.global_member WHERE member_nama ILIKE '%Budi%' AND deleted_at IS NULL LIMIT 10"),
        ("Berapa member laki-laki?",
         "SELECT COUNT(*) AS total FROM global.global_member WHERE member_jenis_kelamin = 'L' AND deleted_at IS NULL"),
    ],
    "global_auth_user": [
        ("Berapa jumlah user aktif?",
         "SELECT COUNT(*) AS total_user FROM global.global_auth_user WHERE usr_status = '1' AND deleted_at IS NULL"),
        ("Tampilkan daftar user",
         "SELECT usr_name, usr_loginname, usr_status FROM global.global_auth_user WHERE deleted_at IS NULL LIMIT 10"),
        ("Siapa pegawai yang terdaftar?",
         "SELECT usr_name, usr_loginname FROM global.global_auth_user WHERE is_pegawai = '1' AND deleted_at IS NULL LIMIT 10"),
        ("Cari user dengan nama Andi",
         "SELECT usr_name, usr_loginname FROM global.global_auth_user WHERE usr_name ILIKE '%Andi%' AND deleted_at IS NULL LIMIT 10"),
    ],
    "global_departemen": [
        ("Tampilkan daftar departemen",
         "SELECT dep_nama, dep_kota FROM global.global_departemen WHERE dep_aktif = '1' AND deleted_at IS NULL LIMIT 20"),
        ("Berapa jumlah departemen aktif?",
         "SELECT COUNT(*) AS total_departemen FROM global.global_departemen WHERE dep_aktif = '1' AND deleted_at IS NULL"),
        ("Departemen di kota Jakarta",
         "SELECT dep_nama, dep_kota FROM global.global_departemen WHERE dep_kota ILIKE '%Jakarta%' AND deleted_at IS NULL LIMIT 10"),
        ("Daftar cabang aktif",
         "SELECT dep_nama, dep_kota, dep_telepon FROM global.global_departemen WHERE dep_aktif = '1' AND deleted_at IS NULL LIMIT 20"),
    ],
    "global_master_outlet": [
        ("Tampilkan daftar outlet",
         "SELECT outlet_nama, outlet_kota, outlet_telepon FROM global.global_master_outlet WHERE outlet_aktif = '1' LIMIT 10"),
        ("Berapa jumlah outlet?",
         "SELECT COUNT(*) AS total_outlet FROM global.global_master_outlet WHERE outlet_aktif = '1'"),
        ("Outlet di kota Surabaya",
         "SELECT outlet_nama, outlet_alamat, outlet_telepon FROM global.global_master_outlet WHERE outlet_kota ILIKE '%Surabaya%' LIMIT 10"),
    ],
    "global_shift": [
        ("Tampilkan daftar shift",
         "SELECT shift_nama, shift_jam_awal, shift_jam_akhir FROM global.global_shift WHERE shift_aktif = '1' LIMIT 10"),
        ("Berapa jumlah shift?",
         "SELECT COUNT(*) AS total_shift FROM global.global_shift WHERE shift_aktif = '1'"),
    ],
    "global_menu": [
        ("Tampilkan menu navigasi aplikasi",
         "SELECT menu_label, menu_href, menu_type FROM global.global_menu WHERE is_active = true ORDER BY sort_order LIMIT 10"),
        ("Berapa jumlah menu aktif?",
         "SELECT COUNT(*) AS total_menu FROM global.global_menu WHERE is_active = true"),
        ("Daftar fitur aplikasi",
         "SELECT menu_label, menu_code, menu_type FROM global.global_menu WHERE is_active = true ORDER BY sort_order LIMIT 10"),
    ],
    "global_omset": [
        ("Berapa total omset?",
         "SELECT SUM(omset_nominal) AS total_omset FROM global.global_omset"),
        ("Tampilkan data omset",
         "SELECT omset_nominal, omset_nominal_bawah, id_dep FROM global.global_omset LIMIT 10"),
    ],
    "global_system_activity_log": [
        ("Tampilkan log aktivitas terbaru",
         "SELECT username, action, module, status, created_at FROM global.global_system_activity_log ORDER BY created_at DESC LIMIT 10"),
        ("Log error terbaru",
         "SELECT username, action, message, created_at FROM global.global_system_activity_log WHERE severity = 'ERROR' ORDER BY created_at DESC LIMIT 10"),
        ("Aktivitas user Andi",
         "SELECT action, module, status, created_at FROM global.global_system_activity_log WHERE username ILIKE '%Andi%' ORDER BY created_at DESC LIMIT 10"),
    ],
}


def get_few_shot_prompt(question: str) -> str | None:
    """
    Buat prompt few-shot berdasarkan pertanyaan user.
    Returns None jika query tidak didukung (tabel tidak ada di DB).
    """
    q_lower = question.lower()

    # Cek apakah query minta data yang tidak ada di DB
    if any(kw in q_lower for kw in UNSUPPORTED_KEYWORDS):
        return None

    # Kata "menu" + konteks makanan = menu restoran (tidak ada di dev_richz)
    if "menu" in q_lower and any(kw in q_lower for kw in FOOD_CONTEXT_KEYWORDS):
        return None

    # Deteksi tabel yang relevan
    matched_tables = []
    for table, keywords in TABLE_KEYWORDS.items():
        if any(kw in q_lower for kw in keywords):
            matched_tables.append(table)

    if not matched_tables:
        return None  # Tidak ada tabel yang cocok, jangan generate SQL

    # Build prompt
    lines = [
        "You are a PostgreSQL SQL generator.",
        "Database: PostgreSQL, schema: global",
        "Rules: SELECT only, no INSERT/UPDATE/DELETE, LIMIT 10 max, use exact table/column names below.",
        "",
    ]

    # Schema + examples per tabel yang relevan
    TABLE_SCHEMA = {
        "global_member": "Table global.global_member (member_id, member_nama, member_no_hp, member_email, member_alamat, member_jenis_kelamin, member_poin, member_aktif, member_create, deleted_at)",
        "global_auth_user": "Table global.global_auth_user (usr_id, usr_name, usr_loginname, usr_status, is_pegawai, id_dep, deleted_at)",
        "global_departemen": "Table global.global_departemen (dep_id, dep_nama, dep_kota, dep_aktif, dep_telepon, deleted_at)",
        "global_master_outlet": "Table global.global_master_outlet (outlet_id, outlet_nama, outlet_kota, outlet_alamat, outlet_telepon, outlet_aktif)",
        "global_shift": "Table global.global_shift (shift_id, shift_nama, shift_jam_awal, shift_jam_akhir, shift_aktif)",
        "global_menu": "Table global.global_menu (menu_id, menu_label, menu_code, menu_href, menu_type, sort_order, is_active)",
        "global_omset": "Table global.global_omset (omset_id, omset_nominal, omset_nominal_bawah, id_dep)",
        "global_system_activity_log": "Table global.global_system_activity_log (activity_log_id, username, action, module, severity, status, message, created_at)",
    }

    for table in matched_tables[:2]:  # Max 2 tabel agar tidak overflow token
        if table in TABLE_SCHEMA:
            lines.append(TABLE_SCHEMA[table])

    lines.append("")
    lines.append("Examples:")

    # Ambil contoh dari tabel yang relevan
    example_count = 0
    for table in matched_tables[:2]:
        for q_ex, sql_ex in FEW_SHOT_EXAMPLES.get(table, [])[:3]:
            lines.append(f'Q: "{q_ex}"')
            lines.append(f"SQL: {sql_ex}")
            lines.append("")
            example_count += 1
            if example_count >= 4:  # Max 4 contoh total
                break
        if example_count >= 4:
            break

    lines.append(f'Q: "{question}"')
    lines.append("SQL:")

    return "\n".join(lines)


if __name__ == "__main__":
    # Test
    questions = [
        "Berapa jumlah member?",
        "Tampilkan 5 member terbaru",
        "Daftar departemen aktif",
        "Siapa pegawai yang terdaftar?",
        "Log aktivitas terbaru",
    ]
    for q in questions:
        print(f"\n{'='*60}")
        print(f"Q: {q}")
        print(get_few_shot_prompt(q))
