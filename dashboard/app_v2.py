"""Workforce Pell Opportunity Score dashboard (design-review build).
Run from the repository: streamlit run dashboard/app_v2.py
Uses existing saved data and notebook assets; never retrains or rewrites scores.
"""
from pathlib import Path
import math
from functools import lru_cache
from html import escape
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title='Workforce Pell Opportunity Score', layout='wide', initial_sidebar_state='expanded')
REPO = Path(__file__).resolve().parent.parent
if not (REPO / 'data' / 'national_data' / 'readiness_scores.csv').exists():
    REPO = Path.cwd()
DATA = REPO / 'data' / 'national_data' / 'readiness_scores.csv'
NOTEBOOKS = REPO / 'notebooks'
CIP_FAMILIES = {1:'Agricultural and Veterinary Science',3:'Natural Resources',4:'Architecture',5:'Area and Cultural Studies',9:'Communication and Journalism',10:'Communications Technologies',11:'Computer and Information Sciences',12:'Culinary and Personal Services',13:'Education',14:'Engineering',15:'Engineering Technologies',16:'Foreign Languages',19:'Family and Consumer Sciences',22:'Legal Studies',23:'English',24:'Liberal Arts and General Studies',25:'Library Science',26:'Biological Sciences',27:'Mathematics and Statistics',29:'Military Technologies',30:'Interdisciplinary Studies',31:'Parks, Recreation, and Fitness',38:'Philosophy and Religion',39:'Theology',40:'Physical Sciences',41:'Science Technologies',42:'Psychology',43:'Protective Services and Law Enforcement',44:'Public Administration and Social Service',45:'Social Sciences',46:'Construction Trades',47:'Mechanic and Repair Technologies',48:'Precision Production',49:'Transportation and Materials Moving',50:'Visual and Performing Arts',51:'Health Professions',52:'Business and Management',54:'History'}
SORT_OPTIONS = {'Overall score':'COMBINED','Score within subject':'SUBJECT_PERCENTILE','Outcome score':'OUTCOME','Demand score':'DEMAND_SCORE','School completion rate':'COMPLETION_PCT'}
PAGES = ['Explore Programs','Workforce Insights','Project Overview','Models & Explainability','Data Quality & Limitations']

@st.cache_data
def load_data():
    data = pd.read_csv(DATA, low_memory=False)
    data['SUBJECT'] = data['CIP_FAMILY'].map(CIP_FAMILIES).fillna('Other')
    data['SUBJECT_LABEL'] = data['CIP_FAMILY'].astype('Int64').astype(str).str.zfill(2) + ' ' + data['SUBJECT']
    data['OUTCOME'] = data['OUTCOME_PROB'] * 100
    data['COMPLETION_PCT'] = data['SCHOOL_COMPLETION'] * 100
    data['SUBJECT_PERCENTILE'] = data.groupby('CIP_FAMILY')['COMBINED'].rank(pct=True) * 100
    data['SCORED'] = data['TIER'] != 'Not scored'
    return data

def fmt(value, digits=1, suffix=''):
    return 'Unavailable' if pd.isna(value) else f'{value:.{digits}f}{suffix}'

def fmt_percent(value):
    return fmt(value, 1, '%')

@lru_cache(maxsize=2048)
def details_markup(fields):
    """Render fixed program details without an interactive dataframe."""
    rows = ''.join(
        '<tr><td>' + escape(str(label)) + '</td><td>' + escape(str(value)) + '</td></tr>'
        for label, value in fields
    )
    return ('<div class="program-details"><table>'
            '<colgroup><col style="width:37%"><col style="width:63%"></colgroup>'
            '<thead><tr><th>Detail</th><th>Value</th></tr></thead>'
            '<tbody>' + rows + '</tbody></table></div>')

@lru_cache(maxsize=2048)
def card_markup(name, school, state, category, basis, overall, outcome, demand):
    identity = escape(name)
    meta = escape(f'{school} · {state} · {category} · {basis}')
    return (f'<div class="program-line"><div class="program-identity">'
            f'<div class="program-name">{identity}</div><div class="program-meta">{meta}</div></div>'
            f'<div class="program-metric"><span>Overall</span><strong>{overall}</strong></div>'
            f'<div class="program-metric"><span>Predicted outcome</span><strong>{outcome}</strong></div>'
            f'<div class="program-metric"><span>Demand</span><strong>{demand}</strong></div></div>')

def contains(series, term):
    return series.astype('string').str.contains(term, case=False, regex=False, na=False)

def reset_filters():
    defaults = {'filter_basis': 'Both score parts (recommended)',
                'filter_sort': 'Overall score', 'filter_states': [],
                'filter_subjects': [], 'tier_high': True,
                'tier_medium': True, 'tier_low': True,
                'school_query': '', 'program_query': '', 'cip_query': ''}
    st.session_state.update(defaults)
    st.session_state.result_page = 1


def update_selection(index):
    selected = st.session_state.selected
    if st.session_state[f'choose_{index}']:
        if index not in selected:
            if len(selected) < 5:
                selected.append(index)
            else:
                st.session_state[f'choose_{index}'] = False
                st.session_state.selection_notice = 'You can select up to five programs.'
    elif index in selected:
        selected.remove(index)


def remove_selection(index):
    if index in st.session_state.selected:
        st.session_state.selected.remove(index)
    st.session_state[f'choose_{index}'] = False


def clear_selections():
    for index in st.session_state.selected:
        st.session_state[f'choose_{index}'] = False
    st.session_state.selected = []


def reset_result_page():
    st.session_state.result_page = 1



def go_to_page():
    st.session_state.result_page = st.session_state.page_jump


def previous_page():
    st.session_state.result_page -= 1
    st.session_state.page_jump = st.session_state.result_page


def next_page():
    st.session_state.result_page += 1
    st.session_state.page_jump = st.session_state.result_page

def search_input(label, key, help_text):
    # Native text inputs do not guarantee an embedded clear icon in all Streamlit versions.
    st.text_input(label, key=key, help=help_text, placeholder='Search...', on_change=reset_result_page)

def read_results(name):
    path = NOTEBOOKS / name
    if not path.exists():
        st.info(f'Missing research output: notebooks/{name}')
        return None
    result = pd.read_csv(path, index_col=0)
    result.insert(0, 'Model', result.index)
    return result.reset_index(drop=True)

def model_chart(res, title):
    metrics = [x for x in ['f1','roc_auc'] if x in res.columns]
    if metrics:
        long = res.melt(id_vars='Model', value_vars=metrics, var_name='Metric', value_name='Score')
        long['Metric'] = long['Metric'].map({'f1':'F1','roc_auc':'ROC-AUC'})
        st.plotly_chart(px.bar(long, x='Model', y='Score', color='Metric', barmode='group', range_y=[0,1], title=title), use_container_width=True)

def simple_bar(labels, values, title, x_title, horizontal=False, fmt_text=None):
    d = pd.DataFrame({'label':labels,'value':values})
    if horizontal:
        fig = px.bar(d, x='value', y='label', orientation='h', title=title, labels={'value':x_title,'label':''}, text='value')
        fig.update_yaxes(autorange='reversed')
    else:
        fig = px.bar(d, x='label', y='value', title=title, labels={'value':x_title,'label':''}, text='value')
    if fmt_text:
        fig.update_traces(texttemplate=fmt_text)
    st.plotly_chart(fig, use_container_width=True)

# Keep the layout compact without overriding Streamlit's internal text positioning.
st.markdown("""<style>
.block-container {max-width: 1000px; padding-top: 1rem; padding-bottom: 1.5rem;}
.st-key-program_results [data-testid="stVerticalBlock"] {gap: .45rem;}
.st-key-program_results [data-testid="stVerticalBlockBorderWrapper"] {border-radius: 8px;}
/* The keyed card itself is the padded stVerticalBlock (confirmed in DevTools). */
[data-testid="stVerticalBlock"][class*="st-key-card_result_"],
[data-testid="stVerticalBlock"][class*="st-key-card_pinned_"] {padding-block: 8px;}
.st-key-bottom_controls [data-testid="stNumberInput"] {max-width: 105px;}
.st-key-bottom_controls [data-testid="stNumberInput"] button {display:none;}
.st-key-bottom_controls [data-testid="stNumberInput"] input {text-align:center;}
.program-details {width:min(760px,100%);margin-top:7px;overflow-wrap:anywhere;}
.program-details table {width:100%;border-collapse:separate;border-spacing:0;table-layout:fixed;border:1px solid #383838;border-radius:8px;overflow:hidden;font-size:.85rem;line-height:1.35;}
.program-details th,.program-details td {padding:8px 10px;text-align:left;vertical-align:top;border-bottom:1px solid #383838;}
.program-details th:first-child,.program-details td:first-child {border-right:1px solid #383838;}
.program-details th {background:#292827;color:#c9c9c9;font-weight:400;}
.program-details tbody tr:last-child td {border-bottom:0;}
.program-line {display:grid;grid-template-columns:minmax(0,1fr) 78px 110px 78px;gap:12px;align-items:center;padding:2px 0;}
.program-identity {min-width:0;}
.program-name {font-weight:650;font-size:.93rem;line-height:1.3;overflow-wrap:anywhere;}
.program-meta {color:#a7a7a7;font-size:.78rem;line-height:1.3;margin-top:4px;overflow-wrap:anywhere;}
.program-metric {min-width:0;}
.program-metric span {display:block;color:#aaa;font-size:.74rem;white-space:nowrap;}
.program-metric strong {display:block;font-size:.93rem;margin-top:3px;}
@media(max-width:750px) {.program-line{grid-template-columns:minmax(0,1fr) 70px 90px 70px;gap:5px}.program-meta{font-size:.7rem}.program-metric span{font-size:.65rem}}
</style>""", unsafe_allow_html=True)
if not DATA.exists():
    st.error(f'Scores file not found: {DATA}. Place this file in the repository dashboard/ directory.')
    st.stop()
df = load_data()
if 'page' not in st.session_state:
    st.session_state.page = PAGES[0]
if 'selected' not in st.session_state:
    st.session_state.selected = []
if 'result_page' not in st.session_state:
    st.session_state.result_page = 1
if 'page_jump' not in st.session_state:
    st.session_state.page_jump = 1

st.title('Workforce Pell Opportunity Score')
st.caption('A readiness score for certificate programs, built from public data. It is a planning tool, not a Workforce Pell eligibility determination.')
st.segmented_control('Navigation', PAGES, key='page', selection_mode='single', label_visibility='collapsed')
page = st.session_state.page
if page != 'Explore Programs':
    st.markdown('<style>[data-testid="stSidebar"] {display:none !important;} [data-testid="stSidebarCollapsedControl"] {display:none !important;}</style>', unsafe_allow_html=True)

if page == 'Explore Programs':
    st.sidebar.header('Explore Programs · Filters')
    basis = st.sidebar.radio('Programs to include', ['Both score parts (recommended)','All scored programs'], key='filter_basis', on_change=reset_result_page, help='Programs scored on outcome + demand are not directly comparable to outcome-only programs.')
    states = st.sidebar.multiselect('State', sorted(df['STABBR'].dropna().unique()), key='filter_states', on_change=reset_result_page)
    subjects = st.sidebar.multiselect('Subject', sorted(df['SUBJECT_LABEL'].dropna().unique()), key='filter_subjects', on_change=reset_result_page)
    st.sidebar.markdown('**Overall score category**')
    high = st.sidebar.checkbox('High', value=True, key='tier_high', on_change=reset_result_page)
    medium = st.sidebar.checkbox('Medium', value=True, key='tier_medium', on_change=reset_result_page)
    low = st.sidebar.checkbox('Low', value=True, key='tier_low', on_change=reset_result_page)
    tiers = [label for label, checked in [('High',high),('Medium',medium),('Low',low)] if checked]
    with st.sidebar:
        search_input('Search by school','school_query','Matches any part of the school name; not case-sensitive.')
        search_input('Search by program','program_query','Matches any part of the program name; not case-sensitive.')
        search_input('CIP code','cip_query','Matches digits in the program CIP code.')
        sort_label = st.selectbox('Rank by', list(SORT_OPTIONS), key='filter_sort', on_change=reset_result_page)
        st.button('Reset filters', use_container_width=True, on_click=reset_filters)
    f = df.loc[df['SCORED']].copy()
    if basis.startswith('Both'):
        f = f[f['TIER_BASIS'].eq('outcome + demand')]
    if states:
        f = f[f['STABBR'].isin(states)]
    if subjects:
        f = f[f['SUBJECT_LABEL'].isin(subjects)]
    f = f[f['TIER'].isin(tiers)]
    if st.session_state.get('school_query'):
        f = f[contains(f['INSTNM_inst'],st.session_state.school_query)]
    if st.session_state.get('program_query'):
        f = f[contains(f['CIPDESC'],st.session_state.program_query)]
    if st.session_state.get('cip_query'):
        f = f[contains(f['CIPCODE'].astype('Int64').astype(str),st.session_state.cip_query)]
    f = f.sort_values(SORT_OPTIONS[sort_label], ascending=False, na_position='last', kind='stable')
    if basis.startswith('All'):
        st.warning('This view mixes two scoring recipes: outcome + demand and outcome only. Check the scoring basis before comparing programs.')
    filtered_csv = f.to_csv(index=False).encode('utf-8')
    @st.fragment
    def render_explore_results():
        mode_col, export_col = st.columns([2,1],vertical_alignment='bottom')
        with mode_col:
            display_mode = st.segmented_control('Display', ['Standard','Expanded'],default='Standard',key='display_mode')
        with export_col:
            st.download_button('Export filtered CSV',filtered_csv,file_name='readiness_scores_filtered.csv',mime='text/csv',use_container_width=True)
        per_page = int(st.session_state.get('per_page',100))
        total_pages = max(1,math.ceil((~f.index.isin(st.session_state.selected)).sum()/per_page))
        current = max(1,min(st.session_state.result_page,total_pages))
        st.session_state.result_page = current
        if st.session_state.page_jump != current:
            st.session_state.page_jump = current
        st.markdown(f'**{len(f):,} programs matching** · {len(st.session_state.selected)} of 5 pinned')

        if st.session_state.pop('selection_notice', None):
            st.warning('You can select up to five programs.')
        def render_program_card(idx, row, pinned=False):
            # Pinned cards reuse the exact same identity and score layout as results.
            with st.container(border=True, key=f'card_{"pinned" if pinned else "result"}_{idx}'):
                if pinned:
                    info_col, remove_col = st.columns([9.3, .7], vertical_alignment='center', gap='small')
                    with remove_col:
                        st.button('x', key=f'unpin_{idx}', help='Unpin this program',
                                  on_click=remove_selection, args=(idx,), use_container_width=True)
                else:
                    select_col, info_col = st.columns([.32, 9.68], vertical_alignment='center', gap='small')
                    with select_col:
                        st.checkbox('Pin program', value=idx in st.session_state.selected,
                                    key=f'choose_{idx}', label_visibility='collapsed',
                                    on_change=update_selection, args=(idx,))
                with info_col:
                    html = card_markup(
                        str(row['CIPDESC']), str(row['INSTNM_inst']), str(row['STABBR']),
                        str(row['TIER']), str(row['TIER_BASIS']),
                        fmt(row['COMBINED']), fmt_percent(row['OUTCOME']), fmt(row['DEMAND_SCORE']),
                    )
                    st.html(html)
                if display_mode == 'Expanded':
                    detail_fields = [
                        ('Within-subject percentile', fmt(row['SUBJECT_PERCENTILE'])),
                        ('CIP code', str(row['CIPCODE']) if pd.notna(row['CIPCODE']) else 'Unavailable'),
                        ('Metro area', str(row['CBSA Title']) if pd.notna(row['CBSA Title']) else 'Unavailable'),
                        ('School completion (institution-level)', fmt_percent(row['COMPLETION_PCT'])),
                        ('In Colorado training data', 'Yes' if str(row['IN_COLORADO_TRAINING']).lower() == 'true' else 'No'),
                    ]
                    st.html(details_markup(tuple(detail_fields)))

        if st.session_state.selected:
            with st.container(key='pinned_programs'):
                st.markdown(f'**Pinned programs ({len(st.session_state.selected)}/5)**')
                st.caption('Pinned programs stay visible as you browse or filter. Select x to unpin a program.')
                for idx in list(st.session_state.selected):
                    if idx in df.index:
                        render_program_card(idx, df.loc[idx], pinned=True)
                st.button('Clear pins', key='clear_comparison', on_click=clear_selections)
                if len({df.loc[i, 'TIER_BASIS'] for i in st.session_state.selected if i in df.index}) > 1:
                    st.warning('Pinned programs use different scoring bases; their overall scores are not directly comparable.')

        # Show pinned cards once, in the pinned area, rather than duplicating them in results.
        unpinned_results = f.loc[~f.index.isin(st.session_state.selected)]
        start = (current - 1) * per_page
        with st.container(key='program_results'):
            for idx, row in unpinned_results.iloc[start:start + per_page].iterrows():
                render_program_card(idx, row)
        # Page size and page navigation belong below the result cards.
        with st.container(key='bottom_controls'):
            st.caption(f'Page {current} of {total_pages}')
            size_col, prev_col, page_col, next_col, _ = st.columns([1.7,1.3,1.25,1.3,3.8],vertical_alignment='bottom',gap='small')
            with size_col:
                st.number_input('Programs per page', min_value=1, max_value=500, value=100, step=1, key='per_page', on_change=reset_result_page)
            with prev_col:
                st.button('← Previous', disabled=current<=1, use_container_width=True, on_click=previous_page)
            with page_col:
                st.number_input('Page', min_value=1, max_value=total_pages, step=1, key='page_jump', on_change=go_to_page)
            with next_col:
                st.button('Next →', disabled=current>=total_pages, use_container_width=True, on_click=next_page)
        st.caption('Predicted outcome estimates whether a program meets the Colorado earnings and placement-proxy criteria; it is not an observed employment rate, and the probabilities have not been calibrated. Demand is unavailable when local occupation data could not be matched. Predictions for programs used in training may be overly optimistic. School completion is the whole school’s rate and is not part of the score. High, Medium and Low describe relative scores, not Workforce Pell eligibility.')

    render_explore_results()

elif page == 'Workforce Insights':
    st.subheader('Average outcome score by subject')
    st.write('Program subject accounts for much of the variation in predicted outcomes, so comparisons within a subject are generally more informative than across subjects.')
    s = df[df['OUTCOME'].notna()].groupby('SUBJECT_LABEL').agg(programs=('OUTCOME','size'),avg_outcome=('OUTCOME','mean')).reset_index()
    s = s[s['programs']>=50].sort_values('avg_outcome')
    if len(s):
        fig=px.bar(s,x='avg_outcome',y='SUBJECT_LABEL',orientation='h',hover_data=['programs'],labels={'avg_outcome':'Average outcome score (0 to 100)','SUBJECT_LABEL':''})
        fig.update_layout(height=max(400,28*len(s)))
        st.plotly_chart(fig,use_container_width=True)
    st.caption('Subjects with at least 50 scored programs.')
    st.subheader('Median overall score by state')
    both=df[(df['TIER_BASIS']=='outcome + demand')&df['STABBR'].notna()]
    states=both.groupby('STABBR').agg(programs=('COMBINED','size'),median_score=('COMBINED','median')).reset_index()
    states=states[states['programs']>=20]
    if len(states):
        st.plotly_chart(px.choropleth(states,locations='STABBR',locationmode='USA-states',scope='usa',color='median_score',hover_data=['programs'],labels={'median_score':'Median overall score'}),use_container_width=True)
    st.caption("Programs with both score parts, in states with at least 20 such programs. State averages mostly reflect which subjects each state's programs teach.")

elif page == 'Project Overview':
    st.subheader('What we built')
    st.write('The project combines an explainable machine learning model, a program-level readiness score, and a dashboard. Because public data cannot establish Workforce Pell eligibility, the score helps identify certificate programs for further investigation using predicted outcomes and local job demand.')
    st.subheader('Checklist against the project overview')
    checklist=[('Explainable ML models','National and Colorado models, cross-validated by school','Done'),('Accuracy, precision, recall, F1 and ROC-AUC','Evaluation tables','Done'),('SHAP explainability','National and Colorado charts','Done'),('Interpretable Opportunity Score','Saved scores for 41,357 programs','Done, with caveats'),('Regional workforce demand','BLS OEWS; missing for many programs','Partly'),('Interactive dashboard','Streamlit dashboard','In progress'),('Data source quality','Documented missingness and limitations','Done'),('Reproducible workflow','Repository scripts, notebooks and README','Partly'),('Actionable recommendations','Drafts for team and advisor review','Draft'),('Workforce Pell criteria and expert validation','Requires advisor review','Not yet'),('Fairness across institution types','National school size and type analysis','Partly'),('Feature dictionary, technical documentation and final presentation','Feature dictionary is listed in the README; completion of remaining deliverables requires team confirmation','To confirm')]
    st.dataframe(pd.DataFrame(checklist,columns=['Project requirement','What we have','Status']),hide_index=True,use_container_width=True)
    st.subheader('From certificate programs to scored programs')
    funnel=pd.DataFrame({'Stage':['All certificate programs','With school information','With an outcome score','With a demand score','With both scores'],'Programs':[43465,len(df),int(df['OUTCOME'].notna().sum()),int(df['DEMAND_SCORE'].notna().sum()),int((df['OUTCOME'].notna()&df['DEMAND_SCORE'].notna()).sum())]})
    fig=px.bar(funnel,x='Programs',y='Stage',orientation='h',text='Programs');fig.update_yaxes(autorange='reversed');fig.update_traces(texttemplate='%{text:,}');st.plotly_chart(fig,use_container_width=True)
    st.caption('The first count comes from the merged dataset; the remaining counts come from the saved scores file. Some programs lack the information needed for an outcome or demand score.')
    st.subheader('How the pieces fit together')
    st.markdown('1. **Merge** College Scorecard, CIP-to-SOC crosswalk, county and metro files, Census ACS and BLS wages.\n2. **Build two labels:** national school completion of at least 70%, and Colorado Year 1 earnings above the high school baseline with at least 50% of completers found in wage records.\n3. **Train and compare** models using school-grouped cross-validation.\n4. **Explain** the models with SHAP.\n5. **Score** programs with outcome and local demand components; display school completion separately.\n6. **Explore** the results.')
    st.subheader('Draft recommendations')
    st.caption('For team and challenge-advisor review.')
    st.markdown('- **Publish program-level completion and placement.** Missing program-level outcomes prevent us from verifying eligibility.\n- **Compare programs within the same subject and school type.** National rankings can be misleading.\n- **Use the score to shortlist, then verify locally.** Check program length, completion, placement and state approval.\n- **Extend the framework state by state.** Add other states with comparable program-level outcome data.')
    st.subheader('How the score is built')
    st.markdown('- **Predicted outcome:** The Colorado XGBoost model estimates whether graduates’ Year 1 earnings exceed the local high school baseline and at least 50% of completers appear in Colorado wage records. It was trained on 343 programs at 22 schools and applied nationally, only to subjects represented by at least five Colorado programs. These are uncalibrated predictions, not observed employment rates.\n- **Demand:** Average of the percentile ranks for local occupational employment and job concentration. It is unavailable where local data could not be matched.\n- **Overall:** The outcome prediction’s percentile rank averaged with the demand score, when available. Scores are divided into Low, Medium and High categories. Programs without demand data use only the outcome percentile, so the two scoring methods are not directly comparable.\n- **School completion:** Institution-level rate shown separately, not included in the score.')

elif page == 'Models & Explainability':
    st.subheader('Two labels, two sets of models')
    st.write('We evaluated five models on two different outcomes: national school completion and Colorado earnings with a placement proxy. Because the datasets and outcomes differ, the results should not be compared directly.')
    for heading,name,description in [('National model: school completion at 70% or higher','results_national.csv','26,857 programs at 2,789 schools; about 16% labeled high completion.'),('Colorado model: earnings above baseline and placement proxy at least 50%','results_colorado.csv','343 programs at 22 Colorado schools; about 29% labeled positive.')]:
        st.markdown('### '+heading)
        st.write(description)
        res=read_results(name)
        if res is not None:
            model_chart(res,heading+' — F1 and ROC-AUC (5-fold cross-validation grouped by school)')
            cols=[c for c in ['Model','accuracy','precision','recall','f1','roc_auc'] if c in res.columns]
            st.dataframe(res[cols].round(3),hide_index=True,use_container_width=True)
    st.caption('The Dummy model always predicts the most common label. Each Colorado validation fold holds out only four or five schools, so results vary; treat differences of about 0.05 or less cautiously.')
    folds=pd.DataFrame({'Fold':[1,2,3,4,5],'ROC-AUC':[.84,.72,.79,.70,.60],'F1':[.67,.49,.48,.44,.32]}).melt(id_vars='Fold',var_name='Metric',value_name='Score')
    st.plotly_chart(px.bar(folds,x='Fold',y='Score',color='Metric',barmode='group',range_y=[0,1],title='Colorado XGBoost: score by cross-validation fold'),use_container_width=True)
    st.markdown('### What drives the scores')
    left,right=st.columns(2)
    with left:
        p=NOTEBOOKS/'shap_national_beeswarm.png'
        if p.exists(): st.image(str(p),caption='National SHAP: school size is the strongest driver. Large schools push the score down.')
        else: st.info('Missing notebooks/shap_national_beeswarm.png')
    with right:
        p=NOTEBOOKS/'shap_colorado_bar.png'
        if p.exists(): st.image(str(p),caption='Colorado SHAP: program subject is the strongest driver.')
        else: st.info('Missing notebooks/shap_colorado_bar.png')
    st.caption('SHAP shows how features influence model predictions, not what causes outcomes.')
    a,b=st.columns(2)
    with a: simple_bar(['Subject only','All features','Everything except subject'],[.751,.730,.523],'Colorado: ROC-AUC with different inputs','ROC-AUC',fmt_text='%{text:.2f}')
    with b: simple_bar(['School size only','All features','Everything except size'],[.882,.947,.904],'National: ROC-AUC with different inputs','ROC-AUC',fmt_text='%{text:.2f}')
    st.caption('In Colorado, subject alone performs about as well as the full model; without subject, performance is close to chance. Nationally, school size is important, but other features also contain information about school type.')
    co_subj={11:(16,.25),43:(23,.83),15:(21,.62),46:(20,.70),52:(33,.27),48:(21,.19),51:(110,.23),47:(24,.17),13:(17,.06),12:(19,.00)}
    d=pd.DataFrame({'Subject':[CIP_FAMILIES[k]+f' ({v[0]} programs)' for k,v in co_subj.items()],'Share beating baseline':[v[1]*100 for v in co_subj.values()]}).sort_values('Share beating baseline')
    st.plotly_chart(px.bar(d,x='Share beating baseline',y='Subject',orientation='h',title='Colorado: share labeled 1 (earnings above baseline and placement proxy ≥50%)',labels={'Share beating baseline':'Percent of programs','Subject':''}),use_container_width=True)
    st.caption('Subjects with at least 15 Colorado programs.')
    st.subheader('Cross-check with Census PSEO (Colorado)')
    st.write('The Colorado earnings results agree with Census PSEO earnings (rank correlation 0.70). PSEO employment rates also track our placement proxy (rank correlation 0.64), but are higher because they include employment anywhere in the U.S. PSEO does not provide the school-level, four-digit CIP employment rates needed to replace our proxy.')

else:
    st.subheader('How the national label differs by school size and type')
    st.write('The national label (school completion at least 70%) is met more often at small schools and private schools. This may reflect student populations and measurement, not program quality.')
    a,b=st.columns(2)
    with a: simple_bar(['Under 100','100 to 500','500 to 2,000','2,000 to 10,000','Over 10,000'],[65,51,12,1,0],'Programs at or above 70% completion, by school size','Percent of programs',fmt_text='%{text}%')
    with b: simple_bar(['Public','Private for-profit','Private nonprofit'],[10,45,52],'Programs at or above 70% completion, by school type','Percent of programs',fmt_text='%{text}%')
    st.caption('Counts are programs, so schools with many programs weigh more heavily. The completion measure covers first-time, full-time students, not individual programs.')
    st.subheader('Where data went missing')
    gaps=[('Programs with no occupation match','768','1.8%','No CIP-to-SOC crosswalk entry'),('Programs with no school match','2,108','4.9%','903 without school ID; 1,205 schools absent from institution file'),('Schools with no county match','341','5.4%','ZIP not found in HUD crosswalk'),('Schools outside any metro area','614','9.8%','Rural counties outside metro or micropolitan areas'),('Programs with no county economic data','2,395','5.5%','County and metro matching gaps'),('Programs with no local wage match','18,843','43.4%','No occupation/metro match or occupation unreported')]
    st.dataframe(pd.DataFrame(gaps,columns=['Gap','Count','Share','Likely cause']),hide_index=True,use_container_width=True)
    st.caption('Program percentages use 43,465 certificate programs; school-level rows use 6,273 schools. Missing values were retained and flagged.')
    st.subheader('Other limits and interpretation')
    st.markdown('- **School completion is institution-level**, not program-level; it is displayed for context and excluded from the score.\n- **Placement is a proxy:** Colorado wage records miss graduates working out of state, self-employed graduates and those below the wage threshold.\n- **Program length** is rarely available; Workforce Pell eligibility cannot be inferred.\n- **Colorado training data** contains only 343 labeled programs at 22 schools, and national application is unvalidated extrapolation.\n- **Missing local demand:** Some programs lack a wage match and use a different scoring basis.\n- **Calibration:** Predicted probabilities have not been validated as calibrated probabilities.\n- **Comparability:** Score categories represent relative thirds, not eligibility. Subject drives much of the score.')
    st.subheader('Data sources')
    st.markdown('''- [College Scorecard](https://collegescorecard.ed.gov/data/) (including IPEDS-derived fields)
- [Colorado Department of Higher Education](https://cdhe.colorado.gov/return-on-investment-report)
- [NCES CIP-to-SOC crosswalk](https://nces.ed.gov/ipeds/cipcode)
- [BLS Occupational Employment and Wage Statistics](https://www.bls.gov/oes/)
- [Census American Community Survey](https://www.census.gov/programs-surveys/acs/data.html)
- [HUD ZIP-to-county crosswalk](https://www.huduser.gov/portal/datasets/usps_crosswalk.html)
- [Census metro definitions](https://www.census.gov/geographies/reference-files/time-series/demo/metro-micro/delineation-files.html)
- [Census PSEO](https://lehd.ces.census.gov/data/pseo_experimental.html) (Colorado cross-check)''')
    st.subheader('Validation')
    st.write('The repository includes unit tests and validation checks for the national and Colorado datasets. They identify data-quality problems but do not determine Workforce Pell eligibility.')
