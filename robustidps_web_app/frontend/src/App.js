/**
 * RobustIDPS.ai - Main React Application
 * ========================================
 *
 * Real-time intrusion detection dashboard
 *
 * Author: Roger Nick Anaedevha
 */

import React, { useState, useEffect } from 'react';
import {
  AppBar, Toolbar, Typography, Container, Grid, Paper,
  Card, CardContent, Box, Chip, Table, TableBody, TableCell,
  TableContainer, TableHead, TableRow, Alert, CircularProgress
} from '@mui/material';
import {
  Shield, Security, Warning, CheckCircle, Speed, Memory
} from '@mui/icons-material';
import { Line, Doughnut } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  ArcElement,
  Title,
  Tooltip,
  Legend
} from 'chart.js';

// Register ChartJS components
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  ArcElement,
  Title,
  Tooltip,
  Legend
);

const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000/api/v1';
const WS_URL = process.env.REACT_APP_WS_URL || 'ws://localhost:8000/ws';

function App() {
  // State
  const [stats, setStats] = useState({
    total_detections: 0,
    malicious_count: 0,
    benign_count: 0,
    detection_rate: 0.0,
    avg_latency_ms: 0.0
  });

  const [alerts, setAlerts] = useState([]);
  const [recentDetections, setRecentDetections] = useState([]);
  const [loading, setLoading] = useState(true);
  const [websocket, setWebsocket] = useState(null);

  // Fetch initial data
  useEffect(() => {
    fetchStats();
    fetchAlerts();
    fetchRecentDetections();
    setupWebSocket();

    // Polling interval
    const interval = setInterval(() => {
      fetchStats();
      fetchAlerts();
      fetchRecentDetections();
    }, 5000);

    return () => {
      clearInterval(interval);
      if (websocket) {
        websocket.close();
      }
    };
  }, []);

  const fetchStats = async () => {
    try {
      const response = await fetch(`${API_URL}/stats/summary`);
      const data = await response.json();
      setStats(data);
      setLoading(false);
    } catch (error) {
      console.error('Error fetching stats:', error);
    }
  };

  const fetchAlerts = async () => {
    try {
      const response = await fetch(`${API_URL}/alerts?limit=10`);
      const data = await response.json();
      setAlerts(data.items || []);
    } catch (error) {
      console.error('Error fetching alerts:', error);
    }
  };

  const fetchRecentDetections = async () => {
    try {
      const response = await fetch(`${API_URL}/detections?limit=20`);
      const data = await response.json();
      setRecentDetections(data.items || []);
    } catch (error) {
      console.error('Error fetching detections:', error);
    }
  };

  const setupWebSocket = () => {
    const ws = new WebSocket(WS_URL);

    ws.onopen = () => {
      console.log('WebSocket connected');
    };

    ws.onmessage = (event) => {
      const message = JSON.parse(event.data);

      if (message.type === 'alert') {
        setAlerts(prev => [message.data, ...prev.slice(0, 9)]);
      } else if (message.type === 'detection') {
        setRecentDetections(prev => [message.data, ...prev.slice(0, 19)]);
      } else if (message.type === 'stats') {
        setStats(message.data);
      }
    };

    ws.onerror = (error) => {
      console.error('WebSocket error:', error);
    };

    ws.onclose = () => {
      console.log('WebSocket disconnected. Reconnecting...');
      setTimeout(setupWebSocket, 3000);
    };

    setWebsocket(ws);
  };

  // Severity color mapping
  const getSeverityColor = (severity) => {
    const colors = {
      critical: '#d32f2f',
      high: '#f57c00',
      medium: '#fbc02d',
      low: '#388e3c',
      info: '#0288d1'
    };
    return colors[severity] || '#757575';
  };

  // Chart data
  const detectionChartData = {
    labels: ['Malicious', 'Benign'],
    datasets: [{
      data: [stats.malicious_count, stats.benign_count],
      backgroundColor: ['#ef5350', '#66bb6a'],
      borderWidth: 0
    }]
  };

  if (loading) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" minHeight="100vh">
        <CircularProgress size={60} />
      </Box>
    );
  }

  return (
    <Box sx={{ flexGrow: 1, bgcolor: '#f5f5f5', minHeight: '100vh' }}>
      {/* Header */}
      <AppBar position="static" sx={{ bgcolor: '#1a237e' }}>
        <Toolbar>
          <Shield sx={{ mr: 2, fontSize: 32 }} />
          <Typography variant="h5" component="div" sx={{ flexGrow: 1, fontWeight: 'bold' }}>
            RobustIDPS.ai
          </Typography>
          <Typography variant="subtitle1" sx={{ mr: 2 }}>
            Advanced AI-Powered Intrusion Detection
          </Typography>
          <Chip
            icon={<CheckCircle />}
            label="OPERATIONAL"
            color="success"
            variant="outlined"
            sx={{ bgcolor: 'rgba(76, 175, 80, 0.1)' }}
          />
        </Toolbar>
      </AppBar>

      <Container maxWidth="xl" sx={{ mt: 4, mb: 4 }}>
        {/* Statistics Cards */}
        <Grid container spacing={3} sx={{ mb: 4 }}>
          <Grid item xs={12} sm={6} md={3}>
            <Card sx={{ bgcolor: '#1976d2', color: 'white' }}>
              <CardContent>
                <Box display="flex" justifyContent="space-between">
                  <div>
                    <Typography variant="h4" fontWeight="bold">
                      {stats.total_detections.toLocaleString()}
                    </Typography>
                    <Typography variant="body2">Total Detections</Typography>
                  </div>
                  <Security sx={{ fontSize: 48, opacity: 0.3 }} />
                </Box>
              </CardContent>
            </Card>
          </Grid>

          <Grid item xs={12} sm={6} md={3}>
            <Card sx={{ bgcolor: '#d32f2f', color: 'white' }}>
              <CardContent>
                <Box display="flex" justifyContent="space-between">
                  <div>
                    <Typography variant="h4" fontWeight="bold">
                      {stats.malicious_count.toLocaleString()}
                    </Typography>
                    <Typography variant="body2">Threats Detected</Typography>
                  </div>
                  <Warning sx={{ fontSize: 48, opacity: 0.3 }} />
                </Box>
              </CardContent>
            </Card>
          </Grid>

          <Grid item xs={12} sm={6} md={3}>
            <Card sx={{ bgcolor: '#388e3c', color: 'white' }}>
              <CardContent>
                <Box display="flex" justifyContent="space-between">
                  <div>
                    <Typography variant="h4" fontWeight="bold">
                      {(stats.detection_rate * 100).toFixed(1)}%
                    </Typography>
                    <Typography variant="body2">Detection Rate</Typography>
                  </div>
                  <CheckCircle sx={{ fontSize: 48, opacity: 0.3 }} />
                </Box>
              </CardContent>
            </Card>
          </Grid>

          <Grid item xs={12} sm={6} md={3}>
            <Card sx={{ bgcolor: '#7b1fa2', color: 'white' }}>
              <CardContent>
                <Box display="flex" justifyContent="space-between">
                  <div>
                    <Typography variant="h4" fontWeight="bold">
                      {stats.avg_latency_ms.toFixed(0)}ms
                    </Typography>
                    <Typography variant="body2">Avg Latency</Typography>
                  </div>
                  <Speed sx={{ fontSize: 48, opacity: 0.3 }} />
                </Box>
              </CardContent>
            </Card>
          </Grid>
        </Grid>

        {/* Charts and Alerts */}
        <Grid container spacing={3} sx={{ mb: 4 }}>
          {/* Detection Distribution */}
          <Grid item xs={12} md={4}>
            <Paper sx={{ p: 3, height: '100%' }}>
              <Typography variant="h6" gutterBottom>
                Detection Distribution
              </Typography>
              <Box sx={{ height: 250, display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
                <Doughnut data={detectionChartData} options={{ maintainAspectRatio: false }} />
              </Box>
            </Paper>
          </Grid>

          {/* Recent Alerts */}
          <Grid item xs={12} md={8}>
            <Paper sx={{ p: 3, height: '100%' }}>
              <Typography variant="h6" gutterBottom>
                Recent Alerts
              </Typography>
              <Box sx={{ maxHeight: 300, overflow: 'auto' }}>
                {alerts.length === 0 ? (
                  <Alert severity="info">No active alerts</Alert>
                ) : (
                  alerts.map((alert, index) => (
                    <Alert
                      key={index}
                      severity={alert.severity === 'critical' ? 'error' : alert.severity === 'high' ? 'warning' : 'info'}
                      sx={{ mb: 1 }}
                    >
                      <strong>{alert.title}</strong> - {alert.description}
                      <Typography variant="caption" display="block">
                        {new Date(alert.created_at).toLocaleString()}
                      </Typography>
                    </Alert>
                  ))
                )}
              </Box>
            </Paper>
          </Grid>
        </Grid>

        {/* Recent Detections Table */}
        <Paper sx={{ p: 3 }}>
          <Typography variant="h6" gutterBottom>
            Recent Detections
          </Typography>
          <TableContainer>
            <Table size="small">
              <TableHead>
                <TableRow>
                  <TableCell><strong>Timestamp</strong></TableCell>
                  <TableCell><strong>Source IP</strong></TableCell>
                  <TableCell><strong>Destination IP</strong></TableCell>
                  <TableCell><strong>Attack Type</strong></TableCell>
                  <TableCell><strong>Confidence</strong></TableCell>
                  <TableCell><strong>Severity</strong></TableCell>
                  <TableCell><strong>Status</strong></TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {recentDetections.length === 0 ? (
                  <TableRow>
                    <TableCell colSpan={7} align="center">
                      No detections yet
                    </TableCell>
                  </TableRow>
                ) : (
                  recentDetections.map((detection, index) => (
                    <TableRow key={index} sx={{ '&:hover': { bgcolor: '#f5f5f5' } }}>
                      <TableCell>
                        {new Date(detection.detected_at).toLocaleString()}
                      </TableCell>
                      <TableCell>{detection.traffic?.src_ip || 'N/A'}</TableCell>
                      <TableCell>{detection.traffic?.dst_ip || 'N/A'}</TableCell>
                      <TableCell>
                        <Chip
                          label={detection.attack_type}
                          size="small"
                          color={detection.is_malicious ? 'error' : 'success'}
                          variant="outlined"
                        />
                      </TableCell>
                      <TableCell>
                        {(detection.confidence * 100).toFixed(1)}%
                      </TableCell>
                      <TableCell>
                        <Chip
                          label={detection.severity}
                          size="small"
                          sx={{
                            bgcolor: getSeverityColor(detection.severity),
                            color: 'white'
                          }}
                        />
                      </TableCell>
                      <TableCell>
                        <Chip
                          label={detection.status}
                          size="small"
                          variant="outlined"
                        />
                      </TableCell>
                    </TableRow>
                  ))
                )}
              </TableBody>
            </Table>
          </TableContainer>
        </Paper>
      </Container>

      {/* Footer */}
      <Box sx={{ bgcolor: '#1a237e', color: 'white', py: 3, mt: 4 }}>
        <Container>
          <Grid container spacing={2}>
            <Grid item xs={12} md={6}>
              <Typography variant="body2">
                © 2024 RobustIDPS.ai - Advanced AI-Powered Intrusion Detection
              </Typography>
              <Typography variant="caption">
                Built by Roger Nick Anaedevha | MEPhI University PhD Dissertation
              </Typography>
            </Grid>
            <Grid item xs={12} md={6} textAlign="right">
              <Typography variant="body2">
                98.4% Accuracy | Sub-100ms Latency | 12.3M Events/sec
              </Typography>
              <Typography variant="caption">
                Neural ODE · Optimal Transport · Encrypted Traffic Analysis
              </Typography>
            </Grid>
          </Grid>
        </Container>
      </Box>
    </Box>
  );
}

export default App;
