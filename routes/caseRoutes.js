const express = require('express');
const router = express.Router();
const Case = require('../database/Case');
const { predictDelay, checkRetrain, getRetrainStatus } = require('../services/mlService');

// In-memory fallback if MongoDB is not yet running
let memoryCases = [];

// 1. Home Page (GET /)
router.get('/', async (req, res) => {
    try {
        let totalCases = 0, highRisk = 0, mediumRisk = 0, lowRisk = 0;

        try {
            [totalCases, highRisk, mediumRisk, lowRisk] = await Promise.all([
                Case.countDocuments(),
                Case.countDocuments({ risk_level: 'High' }),
                Case.countDocuments({ risk_level: 'Medium' }),
                Case.countDocuments({ risk_level: 'Low' })
            ]);
        } catch (dbErr) {
            totalCases = memoryCases.length;
            highRisk = memoryCases.filter(c => c.risk_level === 'High').length;
            mediumRisk = memoryCases.filter(c => c.risk_level === 'Medium').length;
            lowRisk = memoryCases.filter(c => c.risk_level === 'Low').length;
        }

        res.render('listing/Home', {
            title: 'Early Detection of Land Acquisition Delays',
            stats: { totalCases, highRisk, mediumRisk, lowRisk }
        });
    } catch (err) {
        console.error(err);
        res.render('listing/Home', { title: 'Early Detection of Land Acquisition Delays', stats: { totalCases: 0, highRisk: 0, mediumRisk: 0, lowRisk: 0 } });
    }
});

// 2. Input Form (GET /predict or GET /listing/main)
router.get('/predict', (req, res) => {
    res.render('listing/main', { title: 'New Land Acquisition Case Assessment' });
});

router.get('/listing/main', (req, res) => {
    res.redirect('/predict');
});

// 3. Process Prediction (POST /predict)
router.post('/predict', async (req, res) => {
    try {
        const body = req.body;
        
        const acres = parseFloat(body.land_area_acres) || 0;
        let hectares = parseFloat(body.land_area_hectares) || 0;
        if (!hectares && acres) {
            hectares = Math.round(acres * 0.404686 * 100) / 100;
        }

        const caseData = {
            case_id: body.case_id || `LA-${Date.now().toString().slice(-6)}`,
            project_name: body.project_name || 'Infrastructure Project',
            state: body.state || 'Uttar Pradesh',
            district_type: body.district_type || 'Rural',
            project_type: body.project_type || 'Railway',
            land_type: body.land_type || 'Agricultural',
            notification_stage: body.notification_stage || '3A issued',
            land_area_hectares: hectares || 2.5,
            land_area_acres: acres || Math.round(hectares * 2.47105 * 100) / 100,
            affected_owner_count: parseInt(body.affected_owner_count) || 10,
            is_multi_village: body.is_multi_village === '1' || body.is_multi_village === 1 ? 1 : 0,
            has_title_dispute: body.has_title_dispute === '1' || body.has_title_dispute === 1 ? 1 : 0,
            has_court_case: body.has_court_case === '1' || body.has_court_case === 1 ? 1 : 0,
            has_objection: body.has_objection === '1' || body.has_objection === 1 ? 1 : 0,
            compensation_estimate_inr_lakh: parseFloat(body.compensation_estimate_inr_lakh) || 100,
            compensation_funding_available: body.compensation_funding_available === '1' || body.compensation_funding_available === 1 ? 1 : 0,
            days_since_case_opened: parseInt(body.days_since_case_opened) || 120,
            land_notified_percent: parseFloat(body.land_notified_percent) || 50,
            award_completed_percent: parseFloat(body.award_completed_percent) || 30,
            compensation_disbursed_percent: parseFloat(body.compensation_disbursed_percent) || 20,
            possession_completed_percent: parseFloat(body.possession_completed_percent) || 10,
        };

        // Call FastAPI ML API
        const mlResult = await predictDelay(caseData);
        const pred = mlResult.data;

        const fullCase = {
            ...caseData,
            delay_probability: pred.delay_probability,
            delay_probability_percent: pred.delay_probability_percent,
            risk_level: pred.risk_level,
            is_delayed_predicted: pred.is_delayed_predicted,
            risk_distribution: pred.risk_distribution || { low: 0, medium: 0, high: 0 },
            top_risk_factors: pred.top_risk_factors || [],
            recommended_preventive_actions: pred.recommended_preventive_actions || [],
            shap_explanations: pred.shap_explanations || [],
            processing_time_ms: pred.processing_time_ms || 0,
            warning: mlResult.warning || null
        };

        // Save into MongoDB or in-memory
        try {
            const savedRecord = await Case.create(fullCase);
            fullCase._id = savedRecord._id;
            
            // Asynchronous threshold trigger for continuous self-retraining (e.g. at 500 cases)
            Case.countDocuments()
                .then(totalCount => checkRetrain(totalCount))
                .catch(err => console.warn('Retrain trigger check error:', err.message));
        } catch (dbErr) {
            console.warn('Saved to in-memory history:', dbErr.message);
            fullCase._id = `mem-${Date.now()}`;
            memoryCases.unshift(fullCase);
        }

        res.render('listing/show', {
            title: `Risk Analysis: ${fullCase.case_id}`,
            caseItem: fullCase
        });
    } catch (err) {
        console.error('Error handling prediction:', err);
        res.status(500).send('Error analyzing project case.');
    }
});

// 4. View History (GET /history) with Fast Pagination & Lean Queries
router.get('/history', async (req, res) => {
    try {
        const page = Math.max(1, parseInt(req.query.page) || 1);
        const limit = 30;
        let cases = [];
        let totalRecords = 0;

        try {
            [totalRecords, cases] = await Promise.all([
                Case.countDocuments(),
                Case.find()
                    .sort({ createdAt: -1 })
                    .skip((page - 1) * limit)
                    .limit(limit)
                    .lean()
            ]);
        } catch (dbErr) {
            totalRecords = memoryCases.length;
            cases = memoryCases.slice((page - 1) * limit, page * limit);
        }

        const totalPages = Math.ceil(totalRecords / limit) || 1;

        res.render('listing/history', {
            title: 'Historical Case Assessments',
            cases,
            currentPage: page,
            totalPages,
            totalRecords
        });
    } catch (err) {
        console.error(err);
        res.render('listing/history', { title: 'Historical Case Assessments', cases: [], currentPage: 1, totalPages: 1, totalRecords: 0 });
    }
});

// 5. View Single Case Detail (GET /history/:id)
router.get('/history/:id', async (req, res) => {
    try {
        let caseItem = null;
        try {
            caseItem = await Case.findById(req.params.id).lean();
        } catch (dbErr) {
            caseItem = memoryCases.find(c => c._id == req.params.id);
        }

        if (!caseItem) {
            return res.redirect('/history');
        }

        res.render('listing/show', {
            title: `Assessment: ${caseItem.case_id}`,
            caseItem: caseItem
        });
    } catch (err) {
        console.error(err);
        res.redirect('/history');
    }
});

// 6. Delete Case (POST /history/:id/delete)
router.post('/history/:id/delete', async (req, res) => {
    try {
        try {
            await Case.findByIdAndDelete(req.params.id);
        } catch (dbErr) {
            memoryCases = memoryCases.filter(c => c._id != req.params.id);
        }
        res.redirect('/history');
    } catch (err) {
        console.error(err);
        res.redirect('/history');
    }
});

// Direct view for latest or sample assessment (GET /show)
router.get('/show', async (req, res) => {
    try {
        let latest = null;
        try {
            latest = await Case.findOne().sort({ createdAt: -1 }).lean();
        } catch (e) {
            latest = memoryCases[0];
        }
        res.render('listing/show', {
            title: latest ? `Latest Assessment: ${latest.case_id}` : 'Sample Assessment',
            caseItem: latest
        });
    } catch (err) {
        res.render('listing/show', { title: 'Assessment', caseItem: null });
    }
});

// ── DASHBOARD ──────────────────────────────────────────────────────────────
router.get('/dashboard', async (req, res) => {
    try {
        let allCases = [];
        try {
            allCases = await Case.find({}, {
                risk_level: 1, delay_probability: 1, state: 1, project_type: 1,
                land_notified_percent: 1, award_completed_percent: 1,
                compensation_disbursed_percent: 1, possession_completed_percent: 1,
                top_risk_factors: 1, case_id: 1, project_name: 1, _id: 1
            }).sort({ createdAt: -1 }).lean();
        } catch (e) {
            allCases = memoryCases;
        }

        const total  = allCases.length;
        const high   = allCases.filter(c => c.risk_level === 'High').length;
        const medium = allCases.filter(c => c.risk_level === 'Medium').length;
        const low    = allCases.filter(c => c.risk_level === 'Low').length;
        const avgProb = total > 0
            ? Math.round((allCases.reduce((s, c) => s + (c.delay_probability || 0), 0) / total) * 100)
            : 0;

        // State-wise risk breakdown
        const stateMap = {};
        allCases.forEach(c => {
            if (!stateMap[c.state]) stateMap[c.state] = { high: 0, medium: 0, low: 0 };
            if (c.risk_level === 'High')   stateMap[c.state].high++;
            else if (c.risk_level === 'Medium') stateMap[c.state].medium++;
            else stateMap[c.state].low++;
        });
        const stateLabels = Object.keys(stateMap);
        const stateChartData = {
            labels: stateLabels,
            high:   stateLabels.map(s => stateMap[s].high),
            medium: stateLabels.map(s => stateMap[s].medium),
            low:    stateLabels.map(s => stateMap[s].low),
        };

        // Trend: last 20 cases delay probability over time
        const recent = [...allCases].reverse().slice(-20);
        const trendChartData = {
            labels: recent.map((c, i) => `#${i + 1}`),
            values: recent.map(c => Math.round((c.delay_probability || 0) * 100)),
            colors: recent.map(c => c.risk_level === 'High' ? '#dc3545' : c.risk_level === 'Medium' ? '#ffc107' : '#198754'),
        };

        // Sector distribution
        const sectorMap = {};
        allCases.forEach(c => { sectorMap[c.project_type] = (sectorMap[c.project_type] || 0) + 1; });
        const sectorChartData = {
            labels: Object.keys(sectorMap),
            values: Object.values(sectorMap),
        };

        // Average milestone progress
        const avg = (arr, key) => arr.length ? Math.round(arr.reduce((s, c) => s + (c[key] || 0), 0) / arr.length) : 0;
        const milestoneData = {
            notified:   avg(allCases, 'land_notified_percent'),
            award:      avg(allCases, 'award_completed_percent'),
            disbursed:  avg(allCases, 'compensation_disbursed_percent'),
            possession: avg(allCases, 'possession_completed_percent'),
        };

        // Most common risk factor keywords
        const factorKeywords = {
            'Court Litigation':       0,
            'Title Dispute':          0,
            'Public Objections':      0,
            'Funding Pending':        0,
            'Possession Lag':         0,
            'Multi-Village Jurisdiction': 0,
        };
        allCases.forEach(c => {
            (c.top_risk_factors || []).forEach(f => {
                if (f && (f.includes('court') || f.includes('Court'))) factorKeywords['Court Litigation']++;
                if (f && (f.includes('title') || f.includes('Title'))) factorKeywords['Title Dispute']++;
                if (f && (f.includes('objection') || f.includes('Objection'))) factorKeywords['Public Objections']++;
                if (f && (f.includes('fund') || f.includes('Fund'))) factorKeywords['Funding Pending']++;
                if (f && (f.includes('possession') || f.includes('Possession'))) factorKeywords['Possession Lag']++;
                if (f && (f.includes('village') || f.includes('Village'))) factorKeywords['Multi-Village Jurisdiction']++;
            });
        });
        const sortedFactors = Object.entries(factorKeywords).sort((a, b) => b[1] - a[1]);
        const factorChartData = {
            labels: sortedFactors.map(f => f[0]),
            values: sortedFactors.map(f => f[1]),
        };

        const highRiskCases = allCases.filter(c => c.risk_level === 'High').slice(0, 8);
        const retrainStatus = await getRetrainStatus();

        res.render('listing/dashboard', {
            title: 'Analytics Dashboard',
            stats: { total, high, medium, low, avgProb },
            stateChartData,
            trendChartData,
            sectorChartData,
            milestoneData,
            factorChartData,
            highRiskCases,
            retrainStatus,
        });
    } catch (err) {
        console.error(err);
        res.status(500).send('Dashboard error');
    }
});

// ── GIS MAP ────────────────────────────────────────────────────────────────
router.get('/map', async (req, res) => {
    try {
        let mapCases = [];
        try {
            mapCases = await Case.find({}, {
                case_id: 1, project_name: 1, state: 1, district_type: 1,
                project_type: 1, risk_level: 1, delay_probability: 1,
                delay_probability_percent: 1, days_since_case_opened: 1,
                top_risk_factors: 1, recommended_preventive_actions: 1, _id: 1
            }).sort({ createdAt: -1 }).lean();
        } catch (e) {
            mapCases = memoryCases;
        }
        res.render('listing/map', { title: 'GIS Risk Map', mapCases });
    } catch (err) {
        console.error(err);
        res.status(500).send('Map error');
    }
});

// ── ALERTS ─────────────────────────────────────────────────────────────────
router.get('/alerts', async (req, res) => {
    try {
        let alerts = [];
        try {
            alerts = await Case.find({ risk_level: 'High' }, {
                case_id: 1, project_name: 1, state: 1, district_type: 1, project_type: 1,
                risk_level: 1, delay_probability: 1, delay_probability_percent: 1,
                top_risk_factors: 1, recommended_preventive_actions: 1, land_area_acres: 1,
                land_area_hectares: 1, days_since_case_opened: 1, _id: 1
            }).sort({ delay_probability: -1 }).limit(60).lean();
        } catch (e) {
            alerts = memoryCases.filter(c => c.risk_level === 'High').slice(0, 60);
        }
        res.render('listing/alerts', { title: `High-Risk Alerts (${alerts.length})`, alerts });
    } catch (err) {
        console.error(err);
        res.status(500).send('Alerts error');
    }
});

module.exports = router;

