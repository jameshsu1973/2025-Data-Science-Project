# Product names mapping
PRODUCT_NAMES = {
    'ind_ahor_fin_ult1': 'Saving Account',
    'ind_aval_fin_ult1': 'Guarantees',
    'ind_cco_fin_ult1': 'Current Accounts',
    'ind_cder_fin_ult1': 'Derivada Account',
    'ind_cno_fin_ult1': 'Payroll Account',
    'ind_ctju_fin_ult1': 'Junior Account',
    'ind_ctma_fin_ult1': 'Más particular Account',
    'ind_ctop_fin_ult1': 'particular Account',
    'ind_ctpp_fin_ult1': 'particular Plus Account',
    'ind_deco_fin_ult1': 'Short-term deposits',
    'ind_deme_fin_ult1': 'Medium-term deposits',
    'ind_dela_fin_ult1': 'Long-term deposits',
    'ind_ecue_fin_ult1': 'e-account',
    'ind_fond_fin_ult1': 'Funds',
    'ind_hip_fin_ult1': 'Mortgage',
    'ind_plan_fin_ult1': 'Pensions',
    'ind_pres_fin_ult1': 'Loans',
    'ind_reca_fin_ult1': 'Taxes',
    'ind_tjcr_fin_ult1': 'Credit Card',
    'ind_valo_fin_ult1': 'Securities',
    'ind_viv_fin_ult1': 'Home Account',
    'ind_nomina_ult1': 'Payroll',
    'ind_nom_pens_ult1': 'Pensions',
    'ind_recibo_ult1': 'Direct Debit'
}

# Target columns (products to predict)
TARGET_COLS = list(PRODUCT_NAMES.keys())

# Define the steps/fields of the form
FORM_FIELDS = [
    {
        'id': 'fecha_dato',
        'label': 'Date of Data Snapshot',
        'type': 'date',
        'help': 'The snapshot date for this record. Format: YYYY-MM-DD. Example: 2016-06-28',
        'required': True
    },
    {
        'id': 'ncodpers',
        'label': 'Customer Code (ID)',
        'type': 'number',
        'help': 'Unique identifier for the customer. Range: 1-999999999. Example: 15889, 1234567',
        'min': 1,
        'max': 999999999,
        'required': True
    },
    {
        'id': 'ind_empleado',
        'label': 'Employee Index',
        'type': 'select',
        'help': 'The type of employee relationship with the bank. (Values seen in training data)',
        'options': [
            {'value': 'N', 'text': 'Not Employee'},
            {'value': 'A', 'text': 'Active Employee'},
            {'value': 'B', 'text': 'Ex-Employee'},
            {'value': 'F', 'text': 'Subsidiary/Affiliate'},
            {'value': 'P', 'text': 'Passive Employee'}
        ],
        'required': True
    },
    {
        'id': 'pais_residencia',
        'label': 'Country of Residence',
        'type': 'select',
        'help': 'Country of residence code. Only countries seen in training data are available.',
        'options': [
            {'value': 'ES', 'text': 'Spain'},
            {'value': 'CA', 'text': 'Canada'},
            {'value': 'CH', 'text': 'Switzerland'},
            {'value': 'CL', 'text': 'Chile'},
            {'value': 'IE', 'text': 'Ireland'}
        ],
        'required': True
    },
    {
        'id': 'sexo',
        'label': 'Gender',
        'type': 'radio',
        'help': 'The customer\'s gender.',
        'options': [
            {'value': 'H', 'text': 'Male (Hombre)'},
            {'value': 'V', 'text': 'Female (Mujer)'}
        ],
        'required': True
    },
    {
        'id': 'age',
        'label': 'Age',
        'type': 'number',
        'help': 'The customer\'s age in years. Range: 18-100. Example: 35, 42',
        'min': 18,
        'max': 100,
        'required': True
    },
    {
        'id': 'fecha_alta',
        'label': 'Date of Joining the Bank',
        'type': 'date',
        'help': 'The date when the customer joined the bank. Format: YYYY-MM-DD. Example: 2015-01-12',
        'required': True
    },
    {
        'id': 'ind_nuevo',
        'label': 'New Customer Indicator',
        'type': 'radio',
        'help': 'Whether the customer is registered in the last 6 months. (Values seen in training data)',
        'options': [
            {'value': '0', 'text': 'No'},
            {'value': '1', 'text': 'Yes'}
        ],
        'required': True
    },
    {
        'id': 'antiguedad',
        'label': 'Customer Seniority (Months)',
        'type': 'number',
        'help': 'Number of months the customer has been with the bank. Range: 0-600. Example: 18 (1.5 years), 60 (5 years)',
        'min': 0,
        'max': 600,
        'required': True
    },
    {
        'id': 'indrel',
        'label': 'Relationship Status with Bank',
        'type': 'radio',
        'help': 'Primary customer if you are currently active, Former customer otherwise.',
        'options': [
            {'value': '1', 'text': 'Primary Customer'},
            {'value': '99', 'text': 'Former Customer'}
        ],
        'required': True
    },
    {
        'id': 'ult_fec_cli_1t',
        'label': 'Last Date as Primary Customer',
        'type': 'date',
        'help': 'The last date the customer was classified as a primary customer (only applies if Relationship Status is "Former").',
        'required': False
    },
    {
        'id': 'indrel_1mes',
        'label': 'Relationship Status Last Month',
        'type': 'radio',
        'help': 'Relationship status of the customer in the previous month. (Values seen in training data)',
        'options': [
            {'value': '1', 'text': 'Primary Customer'},
            {'value': '3', 'text': 'Former Primary Customer'}
        ],
        'required': True
    },
    {
        'id': 'tiprel_1mes',
        'label': 'Type of Relationship Last Month',
        'type': 'radio',
        'help': 'Type of relationship the customer had with the bank in the previous month. (Values seen in training data)',
        'options': [
            {'value': 'A', 'text': 'Active'},
            {'value': 'I', 'text': 'Inactive'}
        ],
        'required': True
    },
    {
        'id': 'indresi',
        'label': 'Resident Indicator',
        'type': 'radio',
        'help': 'Yes if the residence country is the same as the bank country, No otherwise.',
        'options': [
            {'value': 'S', 'text': 'Resident'},
            {'value': 'N', 'text': 'Non-Resident'}
        ],
        'required': True
    },
    {
        'id': 'indext',
        'label': 'Foreigner Indicator',
        'type': 'radio',
        'help': 'Yes if the customer is a foreigner, No otherwise.',
        'options': [
            {'value': 'S', 'text': 'Foreigner'},
            {'value': 'N', 'text': 'Not Foreigner'}
        ],
        'required': True
    },
    {
        'id': 'conyuemp',
        'label': 'Spouse Employee Indicator',
        'type': 'radio',
        'help': 'Whether the customer\'s spouse is an employee of the bank.',
        'options': [
            {'value': 'N', 'text': 'No'},
            {'value': 'S', 'text': 'Yes'}
        ],
        'required': True
    },
    {
        'id': 'canal_entrada',
        'label': 'Channel of Entry',
        'type': 'select',
        'help': 'The channel through which the customer entered the bank.',
        'options': [
            {'value': 'KHE', 'text': 'Branch Office'},
            {'value': 'KFA', 'text': 'Phone Banking'},
            {'value': 'KFC', 'text': 'Internet Banking'},
            {'value': 'KHM', 'text': 'ATM'},
            {'value': 'RED', 'text': 'Other Channels'}
        ],
        'required': True
    },
    {
        'id': 'indfall',
        'label': 'Deceased Indicator',
        'type': 'radio',
        'help': 'Whether the customer is deceased or not.',
        'options': [
            {'value': 'N', 'text': 'Deceased'},
            {'value': 'S', 'text': 'Not Deceased'}
        ],
        'required': True
    },
    {
        'id': 'tipodom',
        'label': 'Type of Address',
        'type': 'radio',
        'help': 'Type of address in bank records.',
        'options': [
            {'value': '1', 'text': 'Primary Address'},
            {'value': '0', 'text': 'Secondary Address'}
        ],
        'required': True
    },
    {
        'id': 'cod_prov',
        'label': 'Province/State Code',
        'type': 'select',
        'help': 'Province code for Spanish provinces. (Values seen in training data)',
        'options': [
            {'value': '1', 'text': '1 - ALAVA'},
            {'value': '2', 'text': '2 - ALBACETE'},
            {'value': '3', 'text': '3 - ALICANTE'},
            {'value': '4', 'text': '4 - ALMERIA'},
            {'value': '5', 'text': '5 - AVILA'},
            {'value': '6', 'text': '6 - BADAJOZ'},
            {'value': '7', 'text': '7 - BALEARS, ILLES'},
            {'value': '8', 'text': '8 - BARCELONA'},
            {'value': '9', 'text': '9 - BURGOS'},
            {'value': '10', 'text': '10 - CACERES'},
            {'value': '11', 'text': '11 - CADIZ'},
            {'value': '12', 'text': '12 - CASTELLON'},
            {'value': '13', 'text': '13 - CIUDAD REAL'},
            {'value': '14', 'text': '14 - CORDOBA'},
            {'value': '15', 'text': '15 - CORUÑA, A'},
            {'value': '16', 'text': '16 - CUENCA'},
            {'value': '17', 'text': '17 - GIRONA'},
            {'value': '18', 'text': '18 - GRANADA'},
            {'value': '19', 'text': '19 - GUADALAJARA'},
            {'value': '20', 'text': '20 - GIPUZKOA'},
            {'value': '21', 'text': '21 - HUELVA'},
            {'value': '22', 'text': '22 - HUESCA'},
            {'value': '23', 'text': '23 - JAEN'},
            {'value': '24', 'text': '24 - LEON'},
            {'value': '25', 'text': '25 - LERIDA'},
            {'value': '26', 'text': '26 - RIOJA, LA'},
            {'value': '27', 'text': '27 - LUGO'},
            {'value': '28', 'text': '28 - MADRID'},
            {'value': '29', 'text': '29 - MALAGA'},
            {'value': '30', 'text': '30 - MURCIA'},
            {'value': '31', 'text': '31 - NAVARRA'},
            {'value': '32', 'text': '32 - OURENSE'},
            {'value': '33', 'text': '33 - ASTURIAS'},
            {'value': '34', 'text': '34 - PALENCIA'},
            {'value': '35', 'text': '35 - PALMAS, LAS'},
            {'value': '36', 'text': '36 - PONTEVEDRA'},
            {'value': '37', 'text': '37 - SALAMANCA'},
            {'value': '38', 'text': '38 - SANTA CRUZ DE TENERIFE'},
            {'value': '39', 'text': '39 - CANTABRIA'},
            {'value': '40', 'text': '40 - SEGOVIA'},
            {'value': '41', 'text': '41 - SEVILLA'},
            {'value': '42', 'text': '42 - SORIA'},
            {'value': '43', 'text': '43 - TARRAGONA'},
            {'value': '44', 'text': '44 - TERUEL'},
            {'value': '45', 'text': '45 - TOLEDO'},
            {'value': '46', 'text': '46 - VALENCIA'},
            {'value': '47', 'text': '47 - VALLADOLID'},
            {'value': '48', 'text': '48 - BIZKAIA'},
            {'value': '49', 'text': '49 - ZAMORA'},
            {'value': '50', 'text': '50 - ZARAGOZA'},
            {'value': '51', 'text': '51 - CEUTA'},
            {'value': '52', 'text': '52 - MELILLA'}
        ],
        'required': True
    },
    {
        'id': 'nomprov',
        'label': 'Province/State Name',
        'type': 'select',
        'help': 'Province name. Must match the province code. (Values seen in training data)',
        'options': [
            {'value': 'ALAVA', 'text': 'ALAVA'},
            {'value': 'ALBACETE', 'text': 'ALBACETE'},
            {'value': 'ALICANTE', 'text': 'ALICANTE'},
            {'value': 'ALMERIA', 'text': 'ALMERIA'},
            {'value': 'ASTURIAS', 'text': 'ASTURIAS'},
            {'value': 'AVILA', 'text': 'AVILA'},
            {'value': 'BADAJOZ', 'text': 'BADAJOZ'},
            {'value': 'BALEARS, ILLES', 'text': 'BALEARS, ILLES'},
            {'value': 'BARCELONA', 'text': 'BARCELONA'},
            {'value': 'BIZKAIA', 'text': 'BIZKAIA'},
            {'value': 'BURGOS', 'text': 'BURGOS'},
            {'value': 'CACERES', 'text': 'CACERES'},
            {'value': 'CADIZ', 'text': 'CADIZ'},
            {'value': 'CANTABRIA', 'text': 'CANTABRIA'},
            {'value': 'CASTELLON', 'text': 'CASTELLON'},
            {'value': 'CEUTA', 'text': 'CEUTA'},
            {'value': 'CIUDAD REAL', 'text': 'CIUDAD REAL'},
            {'value': 'CORDOBA', 'text': 'CORDOBA'},
            {'value': 'CORUÑA, A', 'text': 'CORUÑA, A'},
            {'value': 'CUENCA', 'text': 'CUENCA'},
            {'value': 'GIPUZKOA', 'text': 'GIPUZKOA'},
            {'value': 'GIRONA', 'text': 'GIRONA'},
            {'value': 'GRANADA', 'text': 'GRANADA'},
            {'value': 'GUADALAJARA', 'text': 'GUADALAJARA'},
            {'value': 'HUELVA', 'text': 'HUELVA'},
            {'value': 'HUESCA', 'text': 'HUESCA'},
            {'value': 'JAEN', 'text': 'JAEN'},
            {'value': 'LEON', 'text': 'LEON'},
            {'value': 'LERIDA', 'text': 'LERIDA'},
            {'value': 'LUGO', 'text': 'LUGO'},
            {'value': 'MADRID', 'text': 'MADRID'},
            {'value': 'MALAGA', 'text': 'MALAGA'},
            {'value': 'MELILLA', 'text': 'MELILLA'},
            {'value': 'MURCIA', 'text': 'MURCIA'},
            {'value': 'NAVARRA', 'text': 'NAVARRA'},
            {'value': 'OURENSE', 'text': 'OURENSE'},
            {'value': 'PALENCIA', 'text': 'PALENCIA'},
            {'value': 'PALMAS, LAS', 'text': 'PALMAS, LAS'},
            {'value': 'PONTEVEDRA', 'text': 'PONTEVEDRA'},
            {'value': 'RIOJA, LA', 'text': 'RIOJA, LA'},
            {'value': 'SALAMANCA', 'text': 'SALAMANCA'},
            {'value': 'SANTA CRUZ DE TENERIFE', 'text': 'SANTA CRUZ DE TENERIFE'},
            {'value': 'SEGOVIA', 'text': 'SEGOVIA'},
            {'value': 'SEVILLA', 'text': 'SEVILLA'},
            {'value': 'SORIA', 'text': 'SORIA'},
            {'value': 'TARRAGONA', 'text': 'TARRAGONA'},
            {'value': 'TERUEL', 'text': 'TERUEL'},
            {'value': 'TOLEDO', 'text': 'TOLEDO'},
            {'value': 'VALENCIA', 'text': 'VALENCIA'},
            {'value': 'VALLADOLID', 'text': 'VALLADOLID'},
            {'value': 'ZAMORA', 'text': 'ZAMORA'},
            {'value': 'ZARAGOZA', 'text': 'ZARAGOZA'}
        ],
        'required': True
    },
    {
        'id': 'ind_actividad_cliente',
        'label': 'Customer Activity Index',
        'type': 'radio',
        'help': 'Whether the customer is active or inactive.',
        'options': [
            {'value': '1', 'text': 'Active Customer'},
            {'value': '0', 'text': 'Inactive Customer'}
        ],
        'required': True
    },
    {
        'id': 'renta',
        'label': 'Gross Income (Annual)',
        'type': 'number',
        'help': 'Annual gross income in euros. Range: 0-500,000. Example: 50000 (€50k), 101850 (€101.85k)',
        'min': 0,
        'max': 500000,
        'step': 100,
        'required': True
    },
    {
        'id': 'segmento',
        'label': 'Customer Segment',
        'type': 'select',
        'help': 'The customer\'s segment within the bank.',
        'options': [
            {'value': '01 - TOP', 'text': 'VIP'},
            {'value': '02 - PARTICULARES', 'text': 'Individuals'},
            {'value': '03 - UNIVERSITARIO', 'text': 'College Graduated'}
        ],
        'required': True
    }
]

# Feature columns (what user inputs in the form)
FEATURE_COLS = [element['id'] for element in FORM_FIELDS]