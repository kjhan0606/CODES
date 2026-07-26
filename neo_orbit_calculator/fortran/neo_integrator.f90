module neo_integrator
  use, intrinsic :: iso_c_binding
  implicit none

  integer, parameter :: qp = selected_real_kind(33, 4931)
  integer, parameter :: nbody = 37
  integer(c_int), parameter :: body_ids(nbody) = [ &
       10_c_int, 1_c_int, 2_c_int, 399_c_int, 301_c_int, 4_c_int, &
       5_c_int, 599_c_int, 501_c_int, 502_c_int, 503_c_int, 504_c_int, &
       505_c_int, 506_c_int, 514_c_int, 515_c_int, 516_c_int, &
       6_c_int, 7_c_int, 8_c_int, 9_c_int, &
       2000001_c_int, 2000004_c_int, 2000002_c_int, 2000010_c_int, &
       2000511_c_int, 2000704_c_int, 2000052_c_int, 2000087_c_int, &
       2000015_c_int, 2000003_c_int, 2000016_c_int, 2000107_c_int, &
       2000088_c_int, 2000007_c_int, 2000031_c_int, 2000065_c_int ]
  real(qp), parameter :: au_km = 149597870.700_qp
  real(qp), parameter :: day_s = 86400.0_qp
  real(qp), parameter :: c_km_s = 299792.458_qp
  real(qp), parameter :: solar_pressure = 4.56e-6_qp
  real(qp), parameter :: proton_mass_kg = 1.67262192595e-27_qp
  real(qp), parameter :: julian_century_s = 36525.0_qp * day_s

  interface
    subroutine spkez_c(target, et, ref, abcorr, observer, state, light_time) &
        bind(C, name="spkez_c")
      import :: c_int, c_double, c_char
      integer(c_int), value :: target
      real(c_double), value :: et
      character(kind=c_char), intent(in) :: ref(*)
      character(kind=c_char), intent(in) :: abcorr(*)
      integer(c_int), value :: observer
      real(c_double), intent(out) :: state(6)
      real(c_double), intent(out) :: light_time
    end subroutine spkez_c
  end interface

contains

  integer(c_int) function neo_real_precision() bind(C, name="neo_real_precision")
    neo_real_precision = precision(1.0_qp)
  end function neo_real_precision

  subroutine body_state(body_id, et, state)
    integer(c_int), intent(in) :: body_id
    real(qp), intent(in) :: et
    real(qp), intent(out) :: state(6)
    real(c_double) :: state64(6), light_time
    character(kind=c_char), parameter :: ref(6) = [ &
         'J', '2', '0', '0', '0', c_null_char ]
    character(kind=c_char), parameter :: correction(5) = [ &
         'N', 'O', 'N', 'E', c_null_char ]

    call spkez_c(body_id, real(et, c_double), ref, correction, &
         0_c_int, state64, light_time)
    state = real(state64, qp)
  end subroutine body_state

  pure function norm3(vector) result(length)
    real(qp), intent(in) :: vector(3)
    real(qp) :: length
    length = sqrt(sum(vector * vector))
  end function norm3

  pure function cross3(a, b) result(c)
    real(qp), intent(in) :: a(3), b(3)
    real(qp) :: c(3)
    c = [a(2) * b(3) - a(3) * b(2), &
         a(3) * b(1) - a(1) * b(3), &
         a(1) * b(2) - a(2) * b(1)]
  end function cross3

  pure subroutine pole_vector(et, ra_coeff, dec_coeff, pole)
    real(qp), intent(in) :: et, ra_coeff(3), dec_coeff(3)
    real(qp), intent(out) :: pole(3)
    real(qp) :: centuries, ra, dec, degrees_to_radians

    degrees_to_radians = acos(-1.0_qp) / 180.0_qp
    centuries = et / julian_century_s
    ra = (ra_coeff(1) + ra_coeff(2) * centuries + &
         ra_coeff(3) * centuries**2) * degrees_to_radians
    dec = (dec_coeff(1) + dec_coeff(2) * centuries + &
         dec_coeff(3) * centuries**2) * degrees_to_radians
    pole = [cos(dec) * cos(ra), cos(dec) * sin(ra), sin(dec)]
  end subroutine pole_vector

  pure subroutine legendre_even(degree, sine_latitude, polynomial, derivative)
    integer, intent(in) :: degree
    real(qp), intent(in) :: sine_latitude
    real(qp), intent(out) :: polynomial, derivative
    real(qp) :: s

    s = sine_latitude
    select case (degree)
    case (2)
      polynomial = 0.5_qp * (3.0_qp * s**2 - 1.0_qp)
      derivative = 3.0_qp * s
    case (4)
      polynomial = (35.0_qp * s**4 - 30.0_qp * s**2 + 3.0_qp) / 8.0_qp
      derivative = (140.0_qp * s**3 - 60.0_qp * s) / 8.0_qp
    case (6)
      polynomial = (231.0_qp * s**6 - 315.0_qp * s**4 + &
           105.0_qp * s**2 - 5.0_qp) / 16.0_qp
      derivative = (1386.0_qp * s**5 - 1260.0_qp * s**3 + &
           210.0_qp * s) / 16.0_qp
    case default
      polynomial = 0.0_qp
      derivative = 0.0_qp
    end select
  end subroutine legendre_even

  pure subroutine zonal_acceleration(relative, pole, mu, reference_radius, &
       coefficients, acceleration)
    real(qp), intent(in) :: relative(3), pole(3), mu, reference_radius
    real(qp), intent(in) :: coefficients(3)
    real(qp), intent(out) :: acceleration(3)
    real(qp) :: radius, rhat(3), polehat(3), pole_norm, sine_latitude
    real(qp) :: polynomial, derivative, scale
    integer :: coefficient_index, degree

    acceleration = 0.0_qp
    radius = norm3(relative)
    pole_norm = norm3(pole)
    if (radius <= 0.0_qp .or. pole_norm <= 0.0_qp) return
    rhat = relative / radius
    polehat = pole / pole_norm
    sine_latitude = sum(rhat * polehat)
    do coefficient_index = 1, 3
      degree = 2 * coefficient_index
      if (coefficients(coefficient_index) == 0.0_qp) cycle
      call legendre_even(degree, sine_latitude, polynomial, derivative)
      scale = mu * coefficients(coefficient_index) * &
           reference_radius**degree / radius**(degree + 2)
      acceleration = acceleration + scale * ( &
           ((real(degree + 1, qp) * polynomial + &
           sine_latitude * derivative) * rhat) - derivative * polehat)
    end do
  end subroutine zonal_acceleration

  pure subroutine stumpff(z, c_value, s_value)
    real(qp), intent(in) :: z
    real(qp), intent(out) :: c_value, s_value
    real(qp) :: root

    if (z > 1.0e-8_qp) then
      root = sqrt(z)
      c_value = (1.0_qp - cos(root)) / z
      s_value = (root - sin(root)) / root**3
    else if (z < -1.0e-8_qp) then
      root = sqrt(-z)
      c_value = (cosh(root) - 1.0_qp) / (-z)
      s_value = (sinh(root) - root) / root**3
    else
      c_value = 0.5_qp - z / 24.0_qp + z**2 / 720.0_qp
      s_value = 1.0_qp / 6.0_qp - z / 120.0_qp + z**2 / 5040.0_qp
    end if
  end subroutine stumpff

  pure subroutine kepler_shift_position(position, velocity, dt, mu, shifted)
    real(qp), intent(in) :: position(3), velocity(3), dt, mu
    real(qp), intent(out) :: shifted(3)
    real(qp) :: radius, speed2, radial_velocity, alpha, sqrt_mu
    real(qp) :: anomaly, z, c_value, s_value, function, derivative
    real(qp) :: update, f_value, g_value
    integer :: iteration

    if (dt == 0.0_qp) then
      shifted = position
      return
    end if
    radius = norm3(position)
    speed2 = sum(velocity * velocity)
    radial_velocity = sum(position * velocity) / radius
    alpha = 2.0_qp / radius - speed2 / mu
    sqrt_mu = sqrt(mu)
    if (abs(alpha) > 1.0e-12_qp) then
      anomaly = sqrt_mu * abs(alpha) * dt
    else
      anomaly = sqrt_mu * dt / radius
    end if
    do iteration = 1, 50
      z = alpha * anomaly**2
      call stumpff(z, c_value, s_value)
      function = radius * radial_velocity / sqrt_mu * anomaly**2 * c_value + &
           (1.0_qp - alpha * radius) * anomaly**3 * s_value + &
           radius * anomaly - sqrt_mu * dt
      derivative = radius * radial_velocity / sqrt_mu * anomaly * &
           (1.0_qp - z * s_value) + &
           (1.0_qp - alpha * radius) * anomaly**2 * c_value + radius
      update = function / derivative
      anomaly = anomaly - update
      if (abs(update) < 1.0e-11_qp * max(1.0_qp, abs(anomaly))) exit
    end do
    z = alpha * anomaly**2
    call stumpff(z, c_value, s_value)
    f_value = 1.0_qp - anomaly**2 / radius * c_value
    g_value = dt - anomaly**3 / sqrt_mu * s_value
    shifted = f_value * position + g_value * velocity
  end subroutine kepler_shift_position

  pure function marsden_scale(radius_au, r0, m_value, n_value, k_value, &
       alpha_value) result(scale)
    real(qp), intent(in) :: radius_au, r0, m_value, n_value, k_value
    real(qp), intent(in) :: alpha_value
    real(qp) :: scale, ratio

    ratio = radius_au / r0
    scale = alpha_value * ratio**(-m_value) * &
         (1.0_qp + ratio**n_value)**(-k_value)
  end function marsden_scale

  subroutine dynamical_step_limit(t, target, et0, gm, maximum, limit, ok)
    real(qp), intent(in) :: t, target(6), et0, gm(nbody), maximum
    real(qp), intent(out) :: limit
    logical, intent(out) :: ok
    real(qp) :: state(6), displacement(3), relative_velocity(3)
    real(qp) :: distance, speed, gravity_timescale, crossing_timescale
    integer :: index

    limit = maximum
    ok = .false.
    do index = 1, nbody
      if (gm(index) <= 0.0_qp) cycle
      call body_state(body_ids(index), et0 + t, state)
      displacement = target(1:3) - state(1:3)
      relative_velocity = target(4:6) - state(4:6)
      distance = norm3(displacement)
      speed = norm3(relative_velocity)
      if (distance <= 0.0_qp) return
      gravity_timescale = sqrt(distance**3 / gm(index))
      if (speed > 0.0_qp) then
        crossing_timescale = distance / speed
      else
        crossing_timescale = huge(1.0_qp)
      end if
      limit = min( &
           limit, &
           0.05_qp * gravity_timescale, &
           0.05_qp * crossing_timescale &
      )
    end do
    ok = .true.
  end subroutine dynamical_step_limit

  pure subroutine multibody_1pn_acceleration(target, states, gm, &
       correction, ok)
    real(qp), intent(in) :: target(6), states(6, nbody), gm(nbody)
    real(qp), intent(out) :: correction(3)
    logical, intent(out) :: ok
    real(qp) :: source_acceleration(3, nbody), source_potential(nbody)
    real(qp) :: displacement(3), distance, target_distance(nbody)
    real(qp) :: target_displacement(3, nbody), target_potential
    real(qp) :: target_speed2, source_speed2, projected_velocity
    real(qp) :: projected_acceleration, factor1, factor2
    real(qp) :: relative_velocity(3)
    integer :: source, other

    correction = 0.0_qp
    source_acceleration = 0.0_qp
    source_potential = 0.0_qp
    ok = .false.
    do source = 1, nbody
      if (gm(source) <= 0.0_qp) cycle
      do other = 1, nbody
        if (source == other .or. gm(other) <= 0.0_qp) cycle
        displacement = states(1:3, other) - states(1:3, source)
        distance = norm3(displacement)
        if (distance <= 0.0_qp) return
        source_acceleration(:, source) = source_acceleration(:, source) + &
             gm(other) * displacement / distance**3
        source_potential(source) = source_potential(source) + &
             gm(other) / distance
      end do
    end do

    target_potential = 0.0_qp
    do source = 1, nbody
      if (gm(source) <= 0.0_qp) cycle
      target_displacement(:, source) = target(1:3) - states(1:3, source)
      target_distance(source) = norm3(target_displacement(:, source))
      if (target_distance(source) <= 0.0_qp) return
      target_potential = target_potential + gm(source) / &
           target_distance(source)
    end do

    target_speed2 = sum(target(4:6) * target(4:6))
    do source = 1, nbody
      if (gm(source) <= 0.0_qp) cycle
      displacement = target_displacement(:, source)
      distance = target_distance(source)
      source_speed2 = sum(states(4:6, source) * states(4:6, source))
      projected_velocity = sum(displacement * states(4:6, source))
      projected_acceleration = sum( &
           displacement * source_acceleration(:, source))
      factor1 = (4.0_qp * target_potential + source_potential(source) - &
           target_speed2 - 2.0_qp * source_speed2 + &
           4.0_qp * sum(target(4:6) * states(4:6, source)) + &
           1.5_qp * projected_velocity**2 / distance**2 + &
           0.5_qp * projected_acceleration) / c_km_s**2
      correction = correction + gm(source) * displacement / distance**3 * &
           factor1

      relative_velocity = target(4:6) - states(4:6, source)
      factor2 = sum(displacement * &
           (4.0_qp * target(4:6) - 3.0_qp * states(4:6, source)))
      correction = correction + gm(source) / c_km_s**2 * ( &
           factor2 * relative_velocity / distance**3 + &
           3.5_qp * source_acceleration(:, source) / distance)
    end do
    ok = .true.
  end subroutine multibody_1pn_acceleration

  subroutine derivatives(t, y, et0, gm, zonal_radius, zonal_coefficients, &
       pole_ra, pole_dec, options, parameters, dydt, ok)
    real(qp), intent(in) :: t, y(6), et0, gm(nbody)
    real(qp), intent(in) :: zonal_radius(nbody)
    real(qp), intent(in) :: zonal_coefficients(3, nbody)
    real(qp), intent(in) :: pole_ra(3, nbody), pole_dec(3, nbody)
    real(qp), intent(in) :: parameters(16)
    integer(c_int), intent(in) :: options(6)
    real(qp), intent(out) :: dydt(6)
    logical, intent(out) :: ok
    real(qp) :: acceleration(3), state(6), body_states(6, nbody)
    real(qp) :: displacement(3), distance, relativistic_term(3)
    real(qp) :: sun(6), rvec(3), vvec(3), rhat(3), hhat(3), that(3)
    real(qp) :: radius, hnorm, mu, v2, rv, srp, radial_velocity
    real(qp) :: scale, a1, a2, a3, area_mass, cr
    real(qp) :: density, wind_speed, wind_factor, wind_pressure
    real(qp) :: relative_wind(3), relative_wind_speed
    real(qp) :: pole(3), zonal_term(3), lag_position(3), lag_radius
    integer :: index

    acceleration = 0.0_qp
    sun = 0.0_qp
    body_states = 0.0_qp
    do index = 1, nbody
      if (gm(index) == 0.0_qp) cycle
      call body_state(body_ids(index), et0 + t, state)
      body_states(:, index) = state
      if (index == 1) sun = state
      displacement = state(1:3) - y(1:3)
      distance = norm3(displacement)
      if (distance <= 0.0_qp) then
        ok = .false.
        return
      end if
      acceleration = acceleration + gm(index) * displacement / distance**3
      if (options(4) /= 0_c_int .and. zonal_radius(index) > 0.0_qp) then
        call pole_vector(et0 + t, pole_ra(:, index), pole_dec(:, index), pole)
        call zonal_acceleration(-displacement, pole, gm(index), &
             zonal_radius(index), zonal_coefficients(:, index), zonal_term)
        acceleration = acceleration + zonal_term
      end if
    end do

    rvec = y(1:3) - sun(1:3)
    vvec = y(4:6) - sun(4:6)
    radius = norm3(rvec)
    if (radius <= 0.0_qp) then
      ok = .false.
      return
    end if
    rhat = rvec / radius
    hhat = cross3(rvec, vvec)
    hnorm = norm3(hhat)
    if (hnorm <= 0.0_qp) then
      ok = .false.
      return
    end if
    hhat = hhat / hnorm
    that = cross3(hhat, rhat)
    mu = gm(1)

    if (options(1) /= 0_c_int) then
      if (options(6) /= 0_c_int) then
        call multibody_1pn_acceleration(y, body_states, gm, &
             relativistic_term, ok)
        if (.not. ok) return
        acceleration = acceleration + relativistic_term
      else
        v2 = sum(vvec * vvec)
        rv = sum(rvec * vvec)
        acceleration = acceleration + mu / (c_km_s**2 * radius**3) * &
             ((4.0_qp * mu / radius - v2) * rvec + 4.0_qp * rv * vvec)
      end if
    end if

    area_mass = parameters(1)
    cr = parameters(2)
    density = parameters(3)
    wind_speed = parameters(4)
    wind_factor = parameters(5)
    if (area_mass > 0.0_qp) then
      srp = solar_pressure * cr * area_mass / 1000.0_qp * &
           (au_km / radius)**2
      if (options(2) /= 0_c_int) then
        radial_velocity = sum(vvec * rhat)
        acceleration = acceleration + srp * ( &
             (1.0_qp - radial_velocity / c_km_s) * rhat - &
             vvec / c_km_s)
      else
        acceleration = acceleration + srp * rhat
      end if
      if (options(3) /= 0_c_int) then
        relative_wind = wind_speed * rhat - vvec
        relative_wind_speed = norm3(relative_wind)
        wind_pressure = density * 1.0e6_qp * proton_mass_kg * &
             (wind_speed * 1000.0_qp)**2 * wind_factor
        acceleration = acceleration + wind_pressure * area_mass / &
             1000.0_qp * (au_km / radius)**2 * &
             relative_wind_speed * relative_wind / wind_speed**2
      end if
    end if

    a1 = parameters(6)
    a2 = parameters(7)
    a3 = parameters(8)
    if (options(5) /= 0_c_int) then
      call kepler_shift_position(rvec, vvec, -parameters(14) * day_s, &
           mu, lag_position)
      lag_radius = norm3(lag_position) / au_km
      scale = marsden_scale(lag_radius, parameters(9), parameters(10), &
           parameters(11), parameters(12), parameters(13))
    else
      scale = (au_km / radius)**2
    end if
    scale = scale * au_km / day_s**2
    acceleration = acceleration + scale * (a1 * rhat + a2 * that + a3 * hhat)

    dydt(1:3) = y(4:6)
    dydt(4:6) = acceleration
    ok = .true.
  end subroutine derivatives

  subroutine rkdp54_step(t, y, h, et0, gm, zonal_radius, &
       zonal_coefficients, pole_ra, pole_dec, options, parameters, &
       y5, error, ok, nfev)
    real(qp), intent(in) :: t, y(6), h, et0, gm(nbody)
    real(qp), intent(in) :: zonal_radius(nbody)
    real(qp), intent(in) :: zonal_coefficients(3, nbody)
    real(qp), intent(in) :: pole_ra(3, nbody), pole_dec(3, nbody)
    real(qp), intent(in) :: parameters(16)
    integer(c_int), intent(in) :: options(6)
    real(qp), intent(out) :: y5(6), error(6)
    logical, intent(out) :: ok
    integer(c_int), intent(inout) :: nfev
    real(qp) :: k1(6), k2(6), k3(6), k4(6), k5(6), k6(6), k7(6)
    real(qp) :: work(6), y4(6)

    call derivatives(t, y, et0, gm, zonal_radius, zonal_coefficients, &
         pole_ra, pole_dec, options, parameters, k1, ok)
    if (.not. ok) return
    work = y + h * (1.0_qp / 5.0_qp) * k1
    call derivatives(t + h / 5.0_qp, work, et0, gm, zonal_radius, &
         zonal_coefficients, pole_ra, pole_dec, options, parameters, k2, ok)
    if (.not. ok) return
    work = y + h * (3.0_qp / 40.0_qp * k1 + 9.0_qp / 40.0_qp * k2)
    call derivatives(t + h * 3.0_qp / 10.0_qp, work, et0, gm, &
         zonal_radius, zonal_coefficients, pole_ra, pole_dec, options, &
         parameters, k3, ok)
    if (.not. ok) return
    work = y + h * (44.0_qp / 45.0_qp * k1 - 56.0_qp / 15.0_qp * k2 + 32.0_qp / 9.0_qp * k3)
    call derivatives(t + h * 4.0_qp / 5.0_qp, work, et0, gm, &
         zonal_radius, zonal_coefficients, pole_ra, pole_dec, options, &
         parameters, k4, ok)
    if (.not. ok) return
    work = y + h * (19372.0_qp / 6561.0_qp * k1 - 25360.0_qp / 2187.0_qp * k2 + &
         64448.0_qp / 6561.0_qp * k3 - 212.0_qp / 729.0_qp * k4)
    call derivatives(t + h * 8.0_qp / 9.0_qp, work, et0, gm, &
         zonal_radius, zonal_coefficients, pole_ra, pole_dec, options, &
         parameters, k5, ok)
    if (.not. ok) return
    work = y + h * (9017.0_qp / 3168.0_qp * k1 - 355.0_qp / 33.0_qp * k2 + &
         46732.0_qp / 5247.0_qp * k3 + 49.0_qp / 176.0_qp * k4 - &
         5103.0_qp / 18656.0_qp * k5)
    call derivatives(t + h, work, et0, gm, zonal_radius, &
         zonal_coefficients, pole_ra, pole_dec, options, parameters, k6, ok)
    if (.not. ok) return
    y5 = y + h * (35.0_qp / 384.0_qp * k1 + 500.0_qp / 1113.0_qp * k3 + &
         125.0_qp / 192.0_qp * k4 - 2187.0_qp / 6784.0_qp * k5 + &
         11.0_qp / 84.0_qp * k6)
    call derivatives(t + h, y5, et0, gm, zonal_radius, &
         zonal_coefficients, pole_ra, pole_dec, options, parameters, k7, ok)
    if (.not. ok) return
    y4 = y + h * (5179.0_qp / 57600.0_qp * k1 + 7571.0_qp / 16695.0_qp * k3 + &
         393.0_qp / 640.0_qp * k4 - 92097.0_qp / 339200.0_qp * k5 + &
         187.0_qp / 2100.0_qp * k6 + 1.0_qp / 40.0_qp * k7)
    error = y5 - y4
    nfev = nfev + 7_c_int
  end subroutine rkdp54_step

  subroutine propagate_neo(initial64, et064, times64, nsample, gm64, &
       zonal_radius64, zonal_coefficients64, pole_ra64, pole_dec64, &
       options, parameters64, rtol64, atol_position64, atol_velocity64, &
       max_step64, output64, nfev, status) bind(C, name="propagate_neo")
    integer(c_int), value :: nsample
    real(c_double), intent(in) :: initial64(6), et064, times64(nsample)
    real(c_double), intent(in) :: gm64(nbody), zonal_radius64(nbody)
    real(c_double), intent(in) :: zonal_coefficients64(3, nbody)
    real(c_double), intent(in) :: pole_ra64(3, nbody), pole_dec64(3, nbody)
    real(c_double), intent(in) :: parameters64(16)
    integer(c_int), intent(in) :: options(6)
    real(c_double), intent(in) :: rtol64, atol_position64, atol_velocity64
    real(c_double), intent(in) :: max_step64
    real(c_double), intent(out) :: output64(6, nsample)
    integer(c_int), intent(out) :: nfev, status
    real(qp) :: y(6), trial(6), error(6), gm(nbody), parameters(16)
    real(qp) :: zonal_radius(nbody), zonal_coefficients(3, nbody)
    real(qp) :: pole_ra(3, nbody), pole_dec(3, nbody)
    real(qp) :: et0, target, t, h, remaining, rtol, atol(6), scale(6)
    real(qp) :: physical_step_limit
    real(qp) :: error_norm, factor, max_step
    logical :: ok
    integer :: index

    y = real(initial64, qp)
    gm = real(gm64, qp)
    zonal_radius = real(zonal_radius64, qp)
    zonal_coefficients = real(zonal_coefficients64, qp)
    pole_ra = real(pole_ra64, qp)
    pole_dec = real(pole_dec64, qp)
    parameters = real(parameters64, qp)
    et0 = real(et064, qp)
    rtol = real(rtol64, qp)
    atol(1:3) = real(atol_position64, qp)
    atol(4:6) = real(atol_velocity64, qp)
    max_step = real(max_step64, qp)
    t = real(times64(1), qp)
    h = min(max_step, max(1.0_qp, (real(times64(nsample), qp) - t) / 1000.0_qp))
    nfev = 0_c_int
    status = 0_c_int
    output64(:, 1) = real(y, c_double)

    do index = 2, nsample
      target = real(times64(index), qp)
      if (target < t) then
        status = 2_c_int
        return
      end if
      do while (t < target)
        remaining = target - t
        call dynamical_step_limit( &
             t, y, et0, gm, max_step, physical_step_limit, ok &
        )
        if (.not. ok) then
          status = 4_c_int
          return
        end if
        h = min(h, remaining, max_step, physical_step_limit)
        if (h <= 1.0e-9_qp) then
          status = 3_c_int
          return
        end if
        call rkdp54_step(t, y, h, et0, gm, zonal_radius, &
             zonal_coefficients, pole_ra, pole_dec, options, parameters, &
             trial, error, ok, nfev)
        if (.not. ok) then
          status = 4_c_int
          return
        end if
        scale = atol + rtol * max(abs(y), abs(trial))
        error_norm = maxval(abs(error) / scale)
        if (error_norm <= 1.0_qp) then
          t = t + h
          y = trial
          if (error_norm == 0.0_qp) then
            factor = 5.0_qp
          else
            factor = min(5.0_qp, max(0.2_qp, 0.9_qp * error_norm**(-0.2_qp)))
          end if
          h = min(max_step, h * factor)
        else
          factor = max(0.1_qp, 0.9_qp * error_norm**(-0.25_qp))
          h = h * factor
        end if
      end do
      output64(:, index) = real(y, c_double)
    end do
  end subroutine propagate_neo

end module neo_integrator
