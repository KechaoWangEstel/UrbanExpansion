




var sample_img = ee.Image('users/tingtinghe2011/00FeasTemp00/sample_1020');
var height_img = generate_result(sample_img);
var vis = {min: 0,  max: 45,  palette: ['blue','green','yellow','orange','red']};
// Map.addLayer(height_img.divide(100), vis, 'height_img');
Map.centerObject(sample_img.geometry(1), 10);
exportImgAsset(height_img, 'result_imgs', 'height_img', null, 30, 'EPSG:4326');



function generate_result(sample_img){
  var roi = sample_img.geometry();
  var all_index_img = get_composite_param_new(roi);
  var train_img_org = all_index_img.addBands(sample_img).updateMask(sample_img.selfMask());
  var list_res = RF_Regression_by_train_img(all_index_img, train_img_org, roi, null, 'height', 30, 500);
  return ee.Image(list_res.get(1)).int().clip(roi);
}
  
function RF_Regression_by_train_img(composite, train_img_org, roi, listBandNames, classPropertyName, scal, treenum){
  classPropertyName = classPropertyName || 'height';
  scal = scal || 30;
  treenum = treenum || 500;
  listBandNames = listBandNames || composite.bandNames();
  
  
  var class_img = get_class_img(train_img_org.select(classPropertyName));
  train_img_org = train_img_org.addBands(class_img);
  var classValues = ee.List.sequence(1, 8, 1);
  var classPoints = ee.List([4000, 3000, 2000, 1000, 800, 600, 400, 200]);
  var trainingPartition = train_img_org.stratifiedSample(1000, 'class', roi, 30, 'EPSG:4326', 1, classValues, classPoints, true, 16, true);
  
  var model = ee.Classifier.smileRandomForest(treenum, null, 5, 0.5, 5, 0)
                      .setOutputMode('REGRESSION')
                      .train({
                              features: trainingPartition,
                              classProperty: classPropertyName,
                              inputProperties: listBandNames});
                              
  var classified = composite.classify(model);
  return ee.List([model, classified]);
}

    
function get_class_img(img){
  var class_img = img.where(img.gt(300).and(img.lte(1000)), 1)
                     .where(img.gt(1000).and(img.lte(2000)), 2)
                     .where(img.gt(2000).and(img.lte(3000)), 3)
                     .where(img.gt(3000).and(img.lte(4000)), 4)
                     .where(img.gt(4000).and(img.lte(5000)), 5)
                     .where(img.gt(5000).and(img.lte(6000)), 6)
                     .where(img.gt(6000).and(img.lte(7000)), 7)
                     .where(img.gt(7000), 8)
                     .rename('class')
  return class_img;
}


{
  function get_composite_param_new(region){
    var startdate='2020-01-01'
    var enddate='2021-12-31'
    function toNatural(img) {
      return ee.Image(10.0).pow(img.divide(10.0));
    }
    var s1_imgs = ee.ImageCollection('COPERNICUS/S1_GRD')
                .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
                .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
                .filter(ee.Filter.eq('instrumentMode', 'IW'))
                .filterDate(startdate,enddate)
                .filterBounds(region)
                .select(['VH','VV'])
                .map(toNatural);
                
    var Sentinel_1_mean= s1_imgs
                .mean()
                .clip(region)
                .rename('Mean_VH','Mean_VV')
    
    var Sentinel_1_max= s1_imgs
                .max()
                .clip(region)
                .rename('Max_VH','Max_VV')
    
    var Sentinel_1_std= s1_imgs
                .reduce(ee.Reducer.stdDev())
                .clip(region)
                .rename('Std_VH','Std_VV')
                
    var VVH_mean=Sentinel_1_mean.expression('VV*5**VH',
        {
          VV: Sentinel_1_mean.select('Mean_VV'),   
          VH: Sentinel_1_mean.select('Mean_VH')
        }).rename('Mean_VVH');
    
    var VVH_max=Sentinel_1_max.expression('VV*5**VH',
        {
          VV: Sentinel_1_max.select('Max_VV'),   
          VH: Sentinel_1_max.select('Max_VH')
        }).rename('Max_VVH');
    
    var VVH_std=Sentinel_1_std.expression('VV*5**VH',
        {
          VV: Sentinel_1_std.select('Std_VV'),   
          VH: Sentinel_1_std.select('Std_VH')
        }).rename('Std_VVH');
        
    var PALSAR=ee.ImageCollection("JAXA/ALOS/PALSAR/YEARLY/SAR")
                 .filterBounds(region)
                 .filterDate('2018-01-01','2022-01-01')
                 .median()
                 .select(['HH','HV'])
                 .multiply(0.0001)
                 .clip(region)
    function maskS2clouds(image) {
      var qa = image.select('QA60');
    
      var cloudBitMask = 1 << 10;
      var cirrusBitMask = 1 << 11;
      var mask = qa.bitwiseAnd(cloudBitMask).eq(0)
          .and(qa.bitwiseAnd(cirrusBitMask).eq(0));
    
      return image.updateMask(mask).divide(10000);
    }
    
    function ND_VI(image,b1,b2,bName)    {
      var VI = image.normalizedDifference([b1,b2]).rename(bName);
      return VI.updateMask(VI.gt(-1).and(VI.lt(1)));
    }
    
    function funEVI(image,B1,B2,B3)    {
      var VI = image.expression('2.5 * (nir - red) / (nir + 6 * red - 7.5 * blue + 1)',
        {
          blue: image.select(B1),   
          red:  image.select(B2),   
          nir:  image.select(B3)
        }).rename('EVI');
      return VI.updateMask(VI.gt(-1).and(VI.lt(1)));
    
    }
    
    function funGCVI(image,B1,B2)    {
    
    	var VI = image.expression('nir/green-1',
        {
          green:  image.select(B1),   
          nir:  image.select(B2)
        }).rename('GCVI');
      return VI.updateMask(VI.gt(-1).and(VI.lt(1)));
    }
    
    function funSEI(image,B1,B2,B3,B4){
      var VI=image.expression('((B1+B2)-(B3+B4))/((B1+B2)+(B3+B4))',
      {
        B1:  image.select(B1),   
        B2:  image.select(B2),   
        B3:  image.select(B3),
        B4:  image.select(B3)
          }).rename('SEI');
      return VI
    }
    function funCSI(img){
      var SEI=funSEI(img,'B1','B9','B3','B8');
      var NDWI=ND_VI(img,'B3','B8','NDWI');
      var NIR=img.select('B8').multiply(0.0001);
      var CSI_1=SEI.subtract(NIR);
      var CSI_2=SEI.subtract(NDWI);
      var mask_1=NIR.gte(NDWI);
      var mask_2=NIR.lt(NDWI);
      return ((CSI_1.multiply(mask_1)).add(CSI_2.multiply(mask_2))).rename('CSI')
    }
    
    function addIndices(img)    {
      var NDVI = ND_VI(img,'B8','B4','NDVI');
      var EVI = funEVI(img,'B1','B3','B4');
      var LSWI = ND_VI(img,'B8','B11','LSWI');
      var mNDWI = ND_VI(img,'B3','B11','mNDWI');
      var GCVI=funGCVI(img,'B8','B3');
      var NDBI=ND_VI(img,'B11','B8','NDBI');
      var CSI=funCSI(img);
      return img.addBands(NDVI).addBands(EVI).addBands(LSWI).addBands(mNDWI).addBands(NDBI).addBands(CSI);
    }
    
    var s2_imgs = ee.ImageCollection("COPERNICUS/S2_SR")
                      .filterDate(startdate,enddate)
                      .filterBounds(region)
                      .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
                      .map(maskS2clouds)
                      .select( 
                      ['B1','B2', 'B3', 'B4','B5','B6','B7', 'B8','B8A','B9', 'B11','B12'])
                      .map(addIndices);
    
    
    var Sentinel_2_mean = s2_imgs
                      .mean()
                      .clip(region)
                      .rename([
                        'Mean_B1',
                        'Mean_B2',
                        'Mean_B3',
                        'Mean_B4',
                        'Mean_B5',
                        'Mean_B6',
                        'Mean_B7',
                        'Mean_B8',
                        'Mean_B8A',
                        'Mean_B9',
                        'Mean_B11',
                        'Mean_B12',
                        'Mean_NDVI',
                        'Mean_EVI',
                        'Mean_LSWI',
                        'Mean_mNDWI',
                        'Mean_NDBI',
                        'Mean_CSI']);
                        
    var Sentinel_2_max = s2_imgs
                      .max()
                      .clip(region)
                      .rename([
                        'Max_B1',
                        'Max_B2',
                        'Max_B3',
                        'Max_B4',
                        'Max_B5',
                        'Max_B6',
                        'Max_B7',
                        'Max_B8',
                        'Max_B8A',
                        'Max_B9',
                        'Max_B11',
                        'Max_B12',
                        'Max_NDVI',
                        'Max_EVI',
                        'Max_LSWI',
                        'Max_mNDWI',
                        'Max_NDBI',
                        'Max_CSI']);
                      
    var Sentinel_2_std = s2_imgs
                      .reduce(ee.Reducer.stdDev())
                      .clip(region)
                      .rename([
                        'Std_B1',
                        'Std_B2',
                        'Std_B3',
                        'Std_B4',
                        'Std_B5',
                        'Std_B6',
                        'Std_B7',
                        'Std_B8',
                        'Std_B8A',
                        'Std_B9',
                        'Std_B11',
                        'Std_B12',
                        'Std_NDVI',
                        'Std_EVI',
                        'Std_LSWI',
                        'Std_mNDWI',
                        'Std_NDBI',
                        'Std_CSI']);
                      
    
    
    var GHSL = ee.ImageCollection("users/ghsl/S2_CNN").mosaic().rename('GHSL').clip(region);
    
    var nightlight = ee.ImageCollection('NOAA/VIIRS/DNB/ANNUAL_V21').select('maximum').filter(ee.Filter.date('2020-01-01', '2021-01-01')).first().rename('nightlight').clip(region);
  
    var Normalize_mean = function(img){
      var kernel = ee.Kernel.circle(50,'meters',false,1)
      var focalmean = img.reduceNeighborhood(ee.Reducer.mean(), kernel)
      return focalmean
    }
    
    var Normalize_max = function(img){
      var kernel = ee.Kernel.circle(50,'meters',false,1)
      var focalmax = img.reduceNeighborhood(ee.Reducer.max(), kernel)
      return focalmax
    }
    
    var data=ee.Image.cat(Sentinel_1_mean,Sentinel_1_max,Sentinel_1_std,
    VVH_mean,VVH_max,VVH_std,PALSAR,
    Sentinel_2_mean,Sentinel_2_max,Sentinel_2_std,
    nightlight,GHSL).reproject('EPSG:4326', null, 10)
    
    var data_mean=Normalize_mean(data)
    var data_max=Normalize_max(data)
    var Data_Merge=ee.Image.cat(data_mean,data_max)
    
    
    var Normalize_sum = function(img){
      var kernel = ee.Kernel.circle(50,'meters',false,1)
      var focalmean = (img.multiply(ee.Image.pixelArea())).reduceNeighborhood(ee.Reducer.sum(), kernel)
      return focalmean
    }
    
    var DEM=ee.Image("CGIAR/SRTM90_V4").reproject('EPSG:4326', null, 10)
    var elevation = DEM.select('elevation');
    var slope = ee.Terrain.slope(elevation);
    var elevation=Normalize_mean(elevation).rename('dem');
    var slope=Normalize_mean(slope).rename('slope');
    
    var wsf2019=ee.ImageCollection('projects/sat-io/open-datasets/WSF/WSF_2019')
    var wsf_01=wsf2019.mosaic().eq(255)
    var buildingup=wsf_01
    var wsf_area=Normalize_sum(wsf_01).rename('wsfarea')
    
    var ESA = ee.ImageCollection("ESA/WorldCover/v100").first().clip(region);
    var Lat_Lon=ESA.addBands(ee.Image.pixelLonLat()).clip(region).select(['longitude','latitude'],['lon','lat'])
    
    var nlt=Normalize_mean(nightlight).rename("NLT_mean");
    var data=ee.Image.cat(Data_Merge,elevation,slope,wsf_area,Lat_Lon,nlt)
                     .reproject('EPSG:4326', null, 10)
    
    var dataforbh=data.select(["HH_mean",
    "HV_mean",
    "Max_B1_max",
    "Max_B1_mean",
    "Max_B2_max",
    "Max_B2_mean",
    "Max_CSI_max",
    "Max_CSI_mean",
    "Max_EVI_max",
    "Max_EVI_mean",
    "Max_B3_max",
    "Max_B3_mean",
    "Max_LSWI_max",
    "Max_LSWI_mean",
    "Max_NDBI_max",
    "Max_NDBI_mean",
    "Max_NDVI_max",
    "Max_NDVI_mean",
    "Max_B8_max",
    "Max_B8_mean",
    "Max_B5_max",
    "Max_B5_mean",
    "Max_B6_max",
    "Max_B6_mean",
    "Max_B7_max",
    "Max_B7_mean",
    "Max_B8A_max",
    "Max_B8A_mean",
    "Max_B4_max",
    "Max_B4_mean",
    "Max_B11_max",
    "Max_B11_mean",
    "Max_B12_max",
    "Max_B12_mean",
    "Max_VH_max",
    "Max_VH_mean",
    "Max_VVH_max",
    "Max_VVH_mean",
    "Max_VV_max",
    "Max_VV_mean",
    "Max_B9_max",
    "Max_B9_mean",
    "Max_mNDWI_max",
    "Max_mNDWI_mean",
    "Mean_B1_max",
    "Mean_B1_mean",
    "Mean_B2_max",
    "Mean_B2_mean",
    "Mean_CSI_max",
    "Mean_CSI_mean",
    "Mean_EVI_max",
    "Mean_EVI_mean",
    "Mean_B3_max",
    "Mean_B3_mean",
    "Mean_LSWI_max",
    "Mean_LSWI_mean",
    "Mean_NDBI_max",
    "Mean_NDBI_mean",
    "Mean_NDVI_max",
    "Mean_NDVI_mean",
    "Mean_B8_max",
    "Mean_B8_mean",
    "Mean_B5_max",
    "Mean_B5_mean",
    "Mean_B6_max",
    "Mean_B6_mean",
    "Mean_B7_max",
    "Mean_B7_mean",
    "Mean_B8A_max",
    "Mean_B8A_mean",
    "Mean_B4_max",
    "Mean_B4_mean",
    "Mean_B11_max",
    "Mean_B11_mean",
    "Mean_B12_max",
    "Mean_B12_mean",
    "Mean_VH_max",
    "Mean_VH_mean",
    "Mean_VVH_max",
    "Mean_VVH_mean",
    "Mean_VV_max",
    "Mean_VV_mean",
    "Mean_B9_max",
    "Mean_B9_mean",
    "Mean_mNDWI_max",
    "Mean_mNDWI_mean",
    "NLT_mean",
    "Std_B1_max",
    "Std_B1_mean",
    "Std_B2_max",
    "Std_B2_mean",
    "Std_CSI_max",
    "Std_CSI_mean",
    "Std_EVI_max",
    "Std_EVI_mean",
    "Std_B3_max",
    "Std_B3_mean",
    "Std_LSWI_max",
    "Std_LSWI_mean",
    "Std_NDBI_max",
    "Std_NDBI_mean",
    "Std_NDVI_max",
    "Std_NDVI_mean",
    "Std_B8_max",
    "Std_B8_mean",
    "Std_B5_max",
    "Std_B5_mean",
    "Std_B6_max",
    "Std_B6_mean",
    "Std_B7_max",
    "Std_B7_mean",
    "Std_B8A_max",
    "Std_B8A_mean",
    "Std_B4_max",
    "Std_B4_mean",
    "Std_B11_max",
    "Std_B11_mean",
    "Std_B12_max",
    "Std_B12_mean",
    "Std_VH_max",
    "Std_VH_mean",
    "Std_VVH_max",
    "Std_VVH_mean",
    "Std_VV_max",
    "Std_VV_mean",
    "Std_B9_max",
    "Std_B9_mean",
    "Std_mNDWI_max",
    "Std_mNDWI_mean",
    "lat",
    "lon",
    "wsfarea",
    "dem",
    "slope"]
    )
    return dataforbh
  }
    
  function Normalize_BH(img, radius){
    var outer = ee.Number(radius)
    var kernel = ee.Kernel.circle(outer,'meters',false,1)
    var B_area=((img.gt(0)).multiply(ee.Image.pixelArea())).reduceNeighborhood(ee.Reducer.sum(), kernel)
    var focalmean = (img.multiply(ee.Image.pixelArea())).reduceNeighborhood(ee.Reducer.sum(), kernel)
    var Nor_img=focalmean.divide(B_area)
    return Nor_img
  }

}


function normalizedDifference(img1, img2) {
  return img1.expression('(img1 - img2) / (img1 + img2)', {
    img1: img1, 
    img2: img2
  });
}


function exportImgAsset(img, foldname, filename, roi_geo, sca, crscrs, pre){
  sca = sca ||30;
  crscrs = crscrs || 'EPSG:4326';
  pre = pre || '';
  filename = filename || 'imgout';
  filename = pre + filename;
  var taskname =  filename;
  filename = foldname + '/' + filename;
  Export.image.toAsset({
    image: ee.Image(img),
    description: taskname, 
    assetId: filename,
    region: roi_geo || img.geometry(),
    scale: sca,
    crs: crscrs|| 'EPSG:4326',
    maxPixels: 1e13
  });
}






