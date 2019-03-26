function tensor_rgb2lab(rgb) {
  let a_con = tf.pow(rgb.add(0.055).div(1.055), 2.4);
  let b_con = rgb.div(12.92);
  let rgb_2 = tf.where(rgb.greater(0.04045), a_con, b_con);

  const [r, g, b] = rgb_2.split(3, 2);

  let x = r
      .mul(0.4124)
      .add(g.mul(0.3576))
      .add(b.mul(0.1805))
      .div(0.95047),
    y = r
      .mul(0.2126)
      .add(g.mul(0.7152))
      .add(b.mul(0.0722))
      .div(1.0),
    z = r
      .mul(0.0193)
      .add(g.mul(0.1192))
      .add(b.mul(0.9505))
      .div(1.08883),
    xyz = tf.concat([x, y, z], 2);

  xyz = tf.where(
    xyz.greater(0.008856),
    tf.pow(xyz, 1 / 3),
    xyz.mul(7.787).add(16 / 116)
  );
  [x, y, z] = xyz.split(3, 2);
  return tf.concat(
    [y.mul(116).sub(16), x.sub(y).mul(500), y.sub(z).mul(200)],
    2
  );
}

function tensor_lab2rgb(lab) {
  let [lab_l, lab_a, lab_b] = lab.split(3, 2),
    y = lab_l.add(16).div(116),
    x = lab_a.div(500).add(y),
    z = y.sub(lab_b.div(200)),
    xyz = tf.concat([x, y, z], 2);

  const xyz_pow3 = xyz.pow(3);

  xyz = tf.where(
    xyz_pow3.greater(0.008856),
    xyz_pow3,
    xyz.sub(16 / 116).div(7.787)
  );

  [x, y, z] = xyz.split(3, 2); // this can be replaced with matmul
  x = x.mul(0.95047);
  y = y.mul(1.0);
  z = z.mul(1.08883);

  let r = x
      .mul(3.2406)
      .add(y.mul(-1.5372))
      .add(z.mul(-0.4986)),
    g = x
      .mul(-0.9689)
      .add(y.mul(1.8758))
      .add(z.mul(0.0415)),
    b = x
      .mul(0.0557)
      .add(y.mul(-0.204))
      .add(z.mul(1.057)),
    rgb = tf.concat([r, g, b], 2);

  rgb = tf.where(
    rgb.greater(0.0031308),
    rgb
      .pow(1 / 2.4)
      .mul(1.055)
      .sub(0.055),
    rgb.mul(12.92)
  );

  return rgb
    .minimum(1)
    .mul(255)
    .maximum(0);
}

function colorize(img_tensor, feat_extractor, model) {
  let resize_tensor = tf.tidy(() => {return tf.image.resizeNearestNeighbor(img_tensor, [224, 224])}),
    // convert img greyscale and extract mobilnet features
    grey_tensor = resize_tensor.div(255).mean(2),
    grey_scale = tf.stack([grey_tensor, grey_tensor, grey_tensor], 2);
    grey_tensor = grey_scale
    .mul(255)
    .div(127.5)
    .sub(1);
    
  const features = tf.tidy(() => {
    return feat_extractor
      .predict(grey_tensor.expandDims())
      .squeeze((axis = [1, 2]));
  });
  // convert greyscale to lab
  let grey_lab = tensor_rgb2lab(grey_scale)
    .split(3, 2)[0]
    .mul(2)
    .div(100)
    .sub(1);
  const ab_pred = tf.tidy(() => {
    return model.predict([grey_lab.expandDims(), features]).squeeze();
  });
  const lab_img = tf.concat([grey_lab.add(1).mul(50), ab_pred.mul(127)], 2);
  return tf.tidy(() => {
    return tensor_lab2rgb(lab_img);
  });
}
