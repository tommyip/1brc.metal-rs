use std::{collections::HashSet, fmt};

const STATION_NAMES: [&'static str; 413] = [
    "Abha",
    "Abidjan",
    "Abéché",
    "Accra",
    "Addis Ababa",
    "Adelaide",
    "Aden",
    "Ahvaz",
    "Albuquerque",
    "Alexandra",
    "Alexandria",
    "Algiers",
    "Alice Springs",
    "Almaty",
    "Amsterdam",
    "Anadyr",
    "Anchorage",
    "Andorra la Vella",
    "Ankara",
    "Antananarivo",
    "Antsiranana",
    "Arkhangelsk",
    "Ashgabat",
    "Asmara",
    "Assab",
    "Astana",
    "Athens",
    "Atlanta",
    "Auckland",
    "Austin",
    "Baghdad",
    "Baguio",
    "Baku",
    "Baltimore",
    "Bamako",
    "Bangkok",
    "Bangui",
    "Banjul",
    "Barcelona",
    "Bata",
    "Batumi",
    "Beijing",
    "Beirut",
    "Belgrade",
    "Belize City",
    "Benghazi",
    "Bergen",
    "Berlin",
    "Bilbao",
    "Birao",
    "Bishkek",
    "Bissau",
    "Blantyre",
    "Bloemfontein",
    "Boise",
    "Bordeaux",
    "Bosaso",
    "Boston",
    "Bouaké",
    "Bratislava",
    "Brazzaville",
    "Bridgetown",
    "Brisbane",
    "Brussels",
    "Bucharest",
    "Budapest",
    "Bujumbura",
    "Bulawayo",
    "Burnie",
    "Busan",
    "Cabo San Lucas",
    "Cairns",
    "Cairo",
    "Calgary",
    "Canberra",
    "Cape Town",
    "Changsha",
    "Charlotte",
    "Chiang Mai",
    "Chicago",
    "Chihuahua",
    "Chișinău",
    "Chittagong",
    "Chongqing",
    "Christchurch",
    "City of San Marino",
    "Colombo",
    "Columbus",
    "Conakry",
    "Copenhagen",
    "Cotonou",
    "Cracow",
    "Da Lat",
    "Da Nang",
    "Dakar",
    "Dallas",
    "Damascus",
    "Dampier",
    "Dar es Salaam",
    "Darwin",
    "Denpasar",
    "Denver",
    "Detroit",
    "Dhaka",
    "Dikson",
    "Dili",
    "Djibouti",
    "Dodoma",
    "Dolisie",
    "Douala",
    "Dubai",
    "Dublin",
    "Dunedin",
    "Durban",
    "Dushanbe",
    "Edinburgh",
    "Edmonton",
    "El Paso",
    "Entebbe",
    "Erbil",
    "Erzurum",
    "Fairbanks",
    "Fianarantsoa",
    "Flores,  Petén",
    "Frankfurt",
    "Fresno",
    "Fukuoka",
    "Gabès",
    "Gaborone",
    "Gagnoa",
    "Gangtok",
    "Garissa",
    "Garoua",
    "George Town",
    "Ghanzi",
    "Gjoa Haven",
    "Guadalajara",
    "Guangzhou",
    "Guatemala City",
    "Halifax",
    "Hamburg",
    "Hamilton",
    "Hanga Roa",
    "Hanoi",
    "Harare",
    "Harbin",
    "Hargeisa",
    "Hat Yai",
    "Havana",
    "Helsinki",
    "Heraklion",
    "Hiroshima",
    "Ho Chi Minh City",
    "Hobart",
    "Hong Kong",
    "Honiara",
    "Honolulu",
    "Houston",
    "Ifrane",
    "Indianapolis",
    "Iqaluit",
    "Irkutsk",
    "Istanbul",
    "İzmir",
    "Jacksonville",
    "Jakarta",
    "Jayapura",
    "Jerusalem",
    "Johannesburg",
    "Jos",
    "Juba",
    "Kabul",
    "Kampala",
    "Kandi",
    "Kankan",
    "Kano",
    "Kansas City",
    "Karachi",
    "Karonga",
    "Kathmandu",
    "Khartoum",
    "Kingston",
    "Kinshasa",
    "Kolkata",
    "Kuala Lumpur",
    "Kumasi",
    "Kunming",
    "Kuopio",
    "Kuwait City",
    "Kyiv",
    "Kyoto",
    "La Ceiba",
    "La Paz",
    "Lagos",
    "Lahore",
    "Lake Havasu City",
    "Lake Tekapo",
    "Las Palmas de Gran Canaria",
    "Las Vegas",
    "Launceston",
    "Lhasa",
    "Libreville",
    "Lisbon",
    "Livingstone",
    "Ljubljana",
    "Lodwar",
    "Lomé",
    "London",
    "Los Angeles",
    "Louisville",
    "Luanda",
    "Lubumbashi",
    "Lusaka",
    "Luxembourg City",
    "Lviv",
    "Lyon",
    "Madrid",
    "Mahajanga",
    "Makassar",
    "Makurdi",
    "Malabo",
    "Malé",
    "Managua",
    "Manama",
    "Mandalay",
    "Mango",
    "Manila",
    "Maputo",
    "Marrakesh",
    "Marseille",
    "Maun",
    "Medan",
    "Mek'ele",
    "Melbourne",
    "Memphis",
    "Mexicali",
    "Mexico City",
    "Miami",
    "Milan",
    "Milwaukee",
    "Minneapolis",
    "Minsk",
    "Mogadishu",
    "Mombasa",
    "Monaco",
    "Moncton",
    "Monterrey",
    "Montreal",
    "Moscow",
    "Mumbai",
    "Murmansk",
    "Muscat",
    "Mzuzu",
    "N'Djamena",
    "Naha",
    "Nairobi",
    "Nakhon Ratchasima",
    "Napier",
    "Napoli",
    "Nashville",
    "Nassau",
    "Ndola",
    "New Delhi",
    "New Orleans",
    "New York City",
    "Ngaoundéré",
    "Niamey",
    "Nicosia",
    "Niigata",
    "Nouadhibou",
    "Nouakchott",
    "Novosibirsk",
    "Nuuk",
    "Odesa",
    "Odienné",
    "Oklahoma City",
    "Omaha",
    "Oranjestad",
    "Oslo",
    "Ottawa",
    "Ouagadougou",
    "Ouahigouya",
    "Ouarzazate",
    "Oulu",
    "Palembang",
    "Palermo",
    "Palm Springs",
    "Palmerston North",
    "Panama City",
    "Parakou",
    "Paris",
    "Perth",
    "Petropavlovsk-Kamchatsky",
    "Philadelphia",
    "Phnom Penh",
    "Phoenix",
    "Pittsburgh",
    "Podgorica",
    "Pointe-Noire",
    "Pontianak",
    "Port Moresby",
    "Port Sudan",
    "Port Vila",
    "Port-Gentil",
    "Portland (OR)",
    "Porto",
    "Prague",
    "Praia",
    "Pretoria",
    "Pyongyang",
    "Rabat",
    "Rangpur",
    "Reggane",
    "Reykjavík",
    "Riga",
    "Riyadh",
    "Rome",
    "Roseau",
    "Rostov-on-Don",
    "Sacramento",
    "Saint Petersburg",
    "Saint-Pierre",
    "Salt Lake City",
    "San Antonio",
    "San Diego",
    "San Francisco",
    "San Jose",
    "San José",
    "San Juan",
    "San Salvador",
    "Sana'a",
    "Santo Domingo",
    "Sapporo",
    "Sarajevo",
    "Saskatoon",
    "Seattle",
    "Ségou",
    "Seoul",
    "Seville",
    "Shanghai",
    "Singapore",
    "Skopje",
    "Sochi",
    "Sofia",
    "Sokoto",
    "Split",
    "St. John's",
    "St. Louis",
    "Stockholm",
    "Surabaya",
    "Suva",
    "Suwałki",
    "Sydney",
    "Tabora",
    "Tabriz",
    "Taipei",
    "Tallinn",
    "Tamale",
    "Tamanrasset",
    "Tampa",
    "Tashkent",
    "Tauranga",
    "Tbilisi",
    "Tegucigalpa",
    "Tehran",
    "Tel Aviv",
    "Thessaloniki",
    "Thiès",
    "Tijuana",
    "Timbuktu",
    "Tirana",
    "Toamasina",
    "Tokyo",
    "Toliara",
    "Toluca",
    "Toronto",
    "Tripoli",
    "Tromsø",
    "Tucson",
    "Tunis",
    "Ulaanbaatar",
    "Upington",
    "Ürümqi",
    "Vaduz",
    "Valencia",
    "Valletta",
    "Vancouver",
    "Veracruz",
    "Vienna",
    "Vientiane",
    "Villahermosa",
    "Vilnius",
    "Virginia Beach",
    "Vladivostok",
    "Warsaw",
    "Washington, D.C.",
    "Wau",
    "Wellington",
    "Whitehorse",
    "Wichita",
    "Willemstad",
    "Winnipeg",
    "Wrocław",
    "Xi'an",
    "Yakutsk",
    "Yangon",
    "Yaoundé",
    "Yellowknife",
    "Yerevan",
    "Yinchuan",
    "Zagreb",
    "Zanzibar City",
    "Zürich",
];

fn name_len_stats() {
    let n_gt_8 = STATION_NAMES.iter().filter(|name| name.len() > 8).count();
    let n_gt_11 = STATION_NAMES.iter().filter(|name| name.len() > 11).count();
    let n_gt_16 = STATION_NAMES.iter().filter(|name| name.len() > 16).count();
    let max_len = STATION_NAMES.iter().map(|name| name.len()).max().unwrap();
    let unicode_prefix = STATION_NAMES
        .iter()
        .filter(|name| name.as_bytes()[0] >> 7 == 1)
        .collect::<Vec<_>>();
    println!(
        ">8={} >11={} >16={} max={}",
        n_gt_8, n_gt_11, n_gt_16, max_len,
    );
    println!("names with unicode prefix: {:?}", unicode_prefix);
}

fn min_prefix() {
    for len in 1.. {
        let n_unique_prefix = STATION_NAMES
            .iter()
            .map(|name| {
                let name = name.as_bytes();
                &name[..name.len().min(len)]
            })
            .collect::<HashSet<_>>()
            .len();
        if n_unique_prefix == STATION_NAMES.len() {
            println!("Minimimum name prefix name: {}", len);
            break;
        }
    }
}

fn djbx33a(s: &[u8]) -> u64 {
    s.iter()
        .fold(5381, |h, c| h.wrapping_mul(33).wrapping_add(*c as u64))
}

fn djbx33a_x4(s: &[u8]) -> u64 {
    let mut chunks = s.chunks_exact(4);
    let mut h = [5381u64, 5381, 5381, 5381];
    while let Some(chunk) = chunks.next() {
        for i in 0..4 {
            h[i] = h[i].wrapping_mul(33).wrapping_add(chunk[i] as u64);
        }
    }
    for (i, &c) in chunks.remainder().iter().enumerate() {
        h[i] = h[i].wrapping_mul(33).wrapping_add(c as u64);
    }
    h[0] ^ h[1] ^ h[2] ^ h[3]
}

fn djbx33a_u64(s: &[u8]) -> u64 {
    let mut h: u64 = 5381;
    for chunk in s.chunks(8) {
        let mut buf = [0u8; 8];
        buf[..chunk.len()].copy_from_slice(chunk);
        h = h.wrapping_mul(33).wrapping_add(u64::from_le_bytes(buf));
    }
    h
}

struct Stats {
    min: u32,
    max: u32,
    avg: f32,
    occupied: usize,
}

impl fmt::Display for Stats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_fmt(format_args!(
            "min: {}, max: {}, avg: {}, occupied: {}",
            self.min, self.max, self.avg, self.occupied
        ))
    }
}

fn statistics<F>(hash_fn: F, hashmap_len: usize) -> Stats
where
    F: Fn(&[u8]) -> u64,
{
    let mut load = vec![0u32; hashmap_len];
    for name in STATION_NAMES {
        let hash = hash_fn(name.as_bytes());
        let idx = hash as usize % hashmap_len;
        load[idx] += 1;
    }
    let occupied_load = load
        .iter()
        .filter(|&&x| x != 0)
        .map(|&x| x)
        .collect::<Vec<_>>();
    let max = *occupied_load.iter().max().unwrap();
    let min = *occupied_load.iter().min().unwrap();
    let avg = occupied_load.iter().sum::<u32>() as f32 / occupied_load.len() as f32;
    Stats {
        max,
        min,
        avg,
        occupied: occupied_load.len(),
    }
}

fn main() {
    let global_len = 10_000;
    let threadgroup_len = 1_365;
    println!("Total names: {}", STATION_NAMES.len());
    name_len_stats();
    min_prefix();

    println!(
        "djbx33a(buckets={}): {}",
        global_len,
        statistics(djbx33a, global_len)
    );
    println!(
        "djbx33a(buckets={}): {}",
        threadgroup_len,
        statistics(djbx33a, threadgroup_len)
    );
    println!(
        "djbx33a_x4(buckets={}): {}",
        global_len,
        statistics(djbx33a_x4, global_len)
    );
    println!(
        "djbx33a_x4(buckets={}): {}",
        threadgroup_len,
        statistics(djbx33a_x4, threadgroup_len)
    );
    println!("djbx33a_u64: {}", djbx33a_u64("abce;".as_bytes()))
}
