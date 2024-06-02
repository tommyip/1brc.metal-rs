#![feature(portable_simd)]

pub mod cpu;
pub mod gpu_baseline;
pub mod opt2;
pub mod opt3;
pub mod opt4;
pub mod test;

use std::{collections::HashMap, env, ffi, fmt, fs::File, mem::size_of};

use memmap2::{Mmap, MmapOptions};
use metal::MTLResourceOptions;

/// Pad end of buffer with null bytes to allow us to read 8/16 bytes
/// at a time for SIMD processing without getting out of bounds
pub const BUF_EXCESS: usize = 16;

pub const U64_SIZE: u64 = size_of::<u64>() as u64;
pub const I32_SIZE: u64 = size_of::<i32>() as u64;
pub const U32_SIZE: u64 = size_of::<u32>() as u64;

pub const STATION_NAMES: [&'static str; 413] = [
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

pub fn mmap<'a, const EXCESS: usize>(file: &'a File) -> (Mmap, usize) {
    let len = file.metadata().unwrap().len() as usize;
    let mmap = unsafe { MmapOptions::new().len(len + EXCESS).map(file).unwrap() };
    (mmap, len)
}

pub fn c_void<T>(value_ref: &T) -> *const ffi::c_void {
    (value_ref as *const T) as *const ffi::c_void
}

#[derive(Copy, Clone)]
pub struct Station {
    min: i16,
    max: i16,
    sum: i32,
    count: u32,
}

impl Default for Station {
    fn default() -> Self {
        Self {
            min: i16::MAX,
            max: i16::MIN,
            sum: 0,
            count: 0,
        }
    }
}

impl Station {
    pub fn new(temp: i16) -> Self {
        Self {
            min: temp,
            max: temp,
            sum: temp as i32,
            count: 1,
        }
    }

    pub fn update(&mut self, temp: i16) {
        if temp < self.min {
            self.min = temp;
        }
        if temp > self.max {
            self.max = temp;
        }
        self.sum += temp as i32;
        self.count += 1;
    }

    pub fn merge(&mut self, other: &Station) {
        if other.min < self.min {
            self.min = other.min;
        }
        if other.max > self.max {
            self.max = other.max;
        }
        self.sum += other.sum;
        self.count += other.count;
    }
}

#[derive(Default)]
pub struct Stations<'a> {
    pub inner: HashMap<&'a [u8], Station>,
}

impl fmt::Display for Station {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let min = self.min as f32 / 10.;
        let max = self.max as f32 / 10.;
        let mean = ((self.sum as f64 / 10. / self.count as f64) * 10.).round() / 10.;
        f.write_fmt(format_args!("{:.1}/{:.1}/{:.1}", min, mean, max))
    }
}

impl<'a> fmt::Display for Stations<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("{")?;
        let mut names = self.inner.keys().collect::<Vec<_>>();
        names.sort_unstable();
        for (i, name) in names.into_iter().enumerate() {
            if i > 0 {
                f.write_str(", ")?;
            }
            let station = &self.inner[name];
            f.write_fmt(format_args!(
                "{}={}",
                unsafe { std::str::from_utf8_unchecked(name) },
                station
            ))?;
        }
        f.write_str("}")
    }
}

pub fn device_buffer<T>(device: &metal::Device, buf: &[T]) -> metal::Buffer {
    device.new_buffer_with_bytes_no_copy(
        buf.as_ptr() as *const ffi::c_void,
        (buf.len() * size_of::<T>()) as u64,
        MTLResourceOptions::StorageModeShared,
        None,
    )
}

pub struct MetalCaptureGuard;

impl Drop for MetalCaptureGuard {
    fn drop(&mut self) {
        metal::CaptureManager::shared().stop_capture();
    }
}

pub fn metal_frame_capture(device: &metal::Device, output_url: &str) -> Option<MetalCaptureGuard> {
    if !env::var("MTL_CAPTURE_ENABLED")
        .ok()
        .is_some_and(|x| &x == "1")
    {
        return None;
    }
    let capture_manager = metal::CaptureManager::shared();
    let capture_descriptor = metal::CaptureDescriptor::new();
    capture_descriptor.set_capture_device(&device);
    capture_descriptor.set_output_url(output_url);
    capture_descriptor.set_destination(metal::MTLCaptureDestination::GpuTraceDocument);

    capture_manager.start_capture(&capture_descriptor).unwrap();

    Some(MetalCaptureGuard)
}

pub fn is_newline(c: &u8) -> bool {
    *c == b'\n'
}
